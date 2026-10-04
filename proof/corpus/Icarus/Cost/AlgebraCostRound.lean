/-
SCOPE: work/proofs/AlgebraCostRound.lean, new, and its receipt work/proofs/AlgebraCostRound.md. A
cost measure for one SHA-256 round of `Apeiron.Crypto.Sha2.compress`, carried over AR-10's
`RingCost σ` with an explicit non-ring layer, the round cost, the schedule-extension step cost, the
64-round chain with its closed form, the per-block figure, and an all-three-faces comparison at
`σ = -1, 0, 1`. Complete when the module elaborates with every headline audited inside
{propext, Classical.choice, Quot.sound} and zero sorryAx, and the receipt records the counts, their
grades, the named premises and the residual list.
REGISTER: algebra/resource model. No production, pin, API, extraction-root, corpus or gate-driver
edit; no Rust source, no cargo, no regen.sh, no measurement. Excluded by construction: latency,
wall-clock time, throughput, bytes, energy, hardware, compiler or codegen cost, the kernel ABI, the
`unsafe` container, device dispatch cost, constant-time and security properties of any hash, and
every asymptotic statement. AGENTS.md:429 - a cost-model proof is not a hardware speed measurement;
this module states counts and nothing else.
CORPUS: `Apeiron.Trichotomy.AnyRing.{RingCell:98,ringMul:109,ringConj:118,ringNormForm:128,
ringMul_comm:180,ringMul_assoc:191}` and
`Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper:139,ringLower:149,ringUpper_mul:160,
ringUpper_fold:509}` at the pinned revision 880aa57e47e782ed7a9d9ee1974cfc87360b446b through
repos/apeiron/proof/corpus/.lake/packages/apeiron/proof/, grade `proved` there
(`Apeiron/Audit.lean:1201` audits `AnyRing.ringUpper`; `:1227` audits `ringUpper_fold`). The round
bodies are `Apeiron.Crypto.Sha2.{rotr:50,shrx:54,sigma0:60,sigma1:64,Sigma0:67,Sigma1:70,
choose:73,majority:76,schedule:82,compress:96}` at repos/apeiron/proof/corpus/Crypto/Sha2.lean,
sha256 b87814767012d1306761c31c78a9c429f4382258201b88899598085e237d5c6e, a root of the
`apeironWordBridge` corpus build (`lakefile.lean:137`). Extracted side:
`proof/corpus/WordRefinement.lean:38` (`RingCell (ZMod (2^64))`), `proof/Denotation.lean:35`
(`denote_wrapping_add`, proved at width 64 only).
REFERENCE: repos/apeiron/AGENTS.md:336-341 (§8 per-function complexity: the measure, the
recurrence, why it closes), :156-167 and :226-240 (§4 obligations and §5 grades), :429 (a
cost-model proof is not a hardware speed measurement).
work/algebra-rooting/ops-map.md:88 (S1 ADD), :91 (S4 XOR), :92 (S5 AND), :94 (S7 ROTR), :95 (S8
SHR), :136-146 (O1 XOR is not `+`, O2 AND is not affine, O3 SHR is not congruence-preserving),
:143 (O6 the word ring does not separate the faces), :167-180 (candidate C5), :220-222 (decisions
D1, D2). work/proofs/AlgebraCost.lean (AR-10) and .md, sha256
52e5d6c5c2b4a144df2a99824052521c29f4b24252d68df5096f7870447df4fe, for `RingCost σ`,
`rep_add`, `chainMulCost`, `rotationCost` and its six-field measure. work/proofs/AlgebraNorm.lean
(AR-7), sha256 4e3d330302a6fd70c5d4736e7eff1add296226893b942a4fca0f8f4efca0f0f7, for the three
mod-8 norm images and the face-separating residue certificates. work/proofs/AlgebraRotate.lean
(AR-9) for `ringUpper_mul_eval`, premise-free because `x.2 * b.2 = 0`.
work/proofs/AlgebraIdempotents.lean (AR-3), sha256
56cb158c97b5f69b40b1c6f149da80189b41e9c61ac801bd9d89b56f8e293bb6, for
`idempotents_trivial_of_pow2`, the idempotents at every dial. receipts/campaign-review.md:30-34,
the owner focus on a formal cost model and on derived costs per sigma face.
GATE: lane lake env lean
~/icarus/work/proofs/AlgebraCostRound.lean, cwd
proof/corpus, the absolute lane wrapper. Wall-clock budget
590 s for the invocation and its controls, including queue time, under
flock /tmp/apeiron/apeiron-lane.lock, MemoryMax=3800M, MemoryHigh=3400M, MemorySwapMax=0,
LEAN_NUM_THREADS=1. No cargo, no lake build of the package, no regen.sh, no nohup/setsid, no
retry loop, no raised maxRecDepth and no maxHeartbeats 0.
GRADE: `proved` for every operation count that is read off a Lean definition body by unfolding it
in this module - the ring bodies through `ringMul`/`ringUpper`/`ringNormForm` and the round bodies
through `Crypto.Sha2` - and for the measure's laws, the chain recurrence and its closure, the
closed forms, the three-face comparison and the elliptic-face root obstruction; the cone is
{propext, Classical.choice, Quot.sound} with no sorryAx, checked by the `#print axioms` rows below.
`model` for two readings and one count: the identification of `ringUpper 1 (ringMul 1 (x,0) (2^n,
2^(32-n)))` with a word rotation (candidate C5, carried as the unused named premise `RotRead`), the
identification of a `Word` addition with a `ZMod (2^32)` addition (the named premise `AddRead`,
template `Denotation.lean:35`, proved at width 64 only), and the count of the seven `Word` additions
of the round body, which is read off the canonical source lines `Sha2.lean:108,109,112,116` because
no machine-checked operation-count mechanism exists. `tested`: none - no `cargo test` ran.
Unmeasured and unclaimed: everything on the REGISTER exclusion list.
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

namespace algebra_cost_round

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 0. The width

The SHA-2 word is `BitVec 32` (`Sha2.lean:22`), and the ring the extracted `Cell<i64>` arithmetic
runs in is `ZMod (2 ^ 64)` (`WordRefinement.lean:38`). A crypto kernel at width `w` is read in
`ZMod (2 ^ w)`; SHA-2 is `w = 32`. Every count that concerns the extracted bodies `word_mul` and
`word_norm` exists only at width 64, so those counts are kept separate from the SHA-2 counts and
never mixed into the round figures of §3. -/

/-- The ring the SHA-2 round is measured in: `w = 32`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- The three dial values of `cell.Sigma` at width 32 (`denoteSigma`, `WordRefinement.lean:29-35`):
elliptic `-1`, parabolic `0`, hyperbolic `1`. -/
abbrev faceElliptic : W32 := -1
/-- The parabolic face's dial value. -/
abbrev faceParabolic : W32 := 0
/-- The hyperbolic face's dial value. -/
abbrev faceHyperbolic : W32 := 1

/-! ## 1. The measure

AR-10's `RingCost σ` has six fields, one counted quantity each. A SHA-256 round additionally
performs operations that have no ring-polynomial reading at word width, so this module states a
record with those operations in **their own fields** rather than folding them into a ring field or
dropping them.

### The eleven counted quantities

| field | counted quantity | why this field exists |
|---|---|---|
| `mul` | scalar ring multiplications | AR-10 `RingCost.mul` |
| `add` | scalar ring additions | AR-10 `RingCost.add` |
| `sub` | scalar ring subtractions | AR-10 `RingCost.sub` |
| `neg` | scalar ring negations | AR-10 `RingCost.neg` |
| `cast` | conversions of a dial value into the ring | AR-10 `RingCost.cast`; paid by the generated bodies, not by the canonical ones |
| `channel` | channel reads, `ringUpper`/`ringLower` applications | AR-10 `RingCost.channel` |
| `bitXor` | bitwise XOR | not ring addition over `ZMod 2^w` (O1) |
| `bitAnd` | bitwise AND | not affine over `ZMod 2^w` (O2) |
| `bitNot` | bitwise NOT | `~~~`, not `neg` over `ZMod 2^w` |
| `shift` | logical shifts, left and right | not congruence-preserving (O3) |
| `bitOr` | bitwise OR | the one disjunct of `rotr` |

A field never mixes two quantities (`CostAlgebra.lean:34-45`), and a record with one field per
quantity has no room for a hidden one: equality is fieldwise equality.

### What the record deliberately excludes

**XOR, AND, NOT and the shifts are counted, not excluded.** The exclusions of O1, O2, O3 and O4 say
these operations are not functions of a word in the ring `ZMod 2^w`; they do not say the operations
do not occur. Dropping them would make the round cost a cost of a *different* function than the
canonical `compress`, and the gap between the two would be invisible. Counting them in their own
fields keeps the claim exact: the ring fields are what the two-slot algebra pays, and the non-ring
fields are what it cannot express.

**Memory, allocation, dispatch and peak live bytes** are excluded because `CostAlgebra.Cost` owns
them. **Element traffic** is excluded and is invisible: `RingCost` has no read field, so the
projections `acc.getD i 0` and `K.getD t 0` and `w.getD t 0` of the round body cost nothing in this
record - the same blindness `CostBridge.loads_invisible_to_order` (`CostBridge.lean:161`) exhibits
for the registered schedule. **Time and every asymptotic claim** are excluded by `AGENTS.md:429`. -/

/-- `RoundCost σ` is the measure: the operation count of one SHA-256 round of `compress`, split into
the ring operations the two-slot algebra `R[x]/(x^2 - sigma)` pays and the word operations it has no
reading for. One counted quantity per field; see §1 for the table. `σ` is the dial value the ring
fields are read at; it enters no count, and §4 states exactly that as a result rather than an
assumption. -/
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
operations, so every count totals: `CostAlgebra.costAdd` (`CostAlgebra.lean:80`) read over ring and
word operations. It is the only composition the module uses. -/
def RoundCost.composed (σ : R) (a b : RoundCost σ) : RoundCost σ :=
  { mul := a.mul + b.mul, add := a.add + b.add, sub := a.sub + b.sub, neg := a.neg + b.neg,
    cast := a.cast + b.cast, channel := a.channel + b.channel, bitXor := a.bitXor + b.bitXor,
    bitAnd := a.bitAnd + b.bitAnd, bitNot := a.bitNot + b.bitNot, shift := a.shift + b.shift,
    bitOr := a.bitOr + b.bitOr }

instance instAddRoundCost (σ : R) : Add (RoundCost σ) := ⟨RoundCost.composed σ⟩

/-- The dependent sum, unfolded, so composition rewrites and its laws close. -/
@[simp] theorem RoundCost.composed_apply (σ : R) (a b : RoundCost σ) :
    a + b = RoundCost.composed σ a b := rfl

/-- **Equality of counts is fieldwise equality.** -/
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

/-- **Composition is commutative**: disjoint work sums the same in either order. -/
theorem RoundCost.composed_comm (σ : R) (a b : RoundCost σ) : a + b = b + a := by
  cases a
  cases b
  simp [RoundCost.composed, Nat.add_comm]

/-- **Composition is associative**: the grouping of a dependent chain does not change its cost,
which is what lets a chain be assembled left- or right-associated. -/
theorem RoundCost.composed_assoc (σ : R) (a b c : RoundCost σ) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [RoundCost.composed, Nat.add_assoc]

/-- **Composition with the zero record is neutral.** -/
theorem RoundCost.composed_zero (σ : R) (a : RoundCost σ) : a + RoundCost.zero σ = a := by
  cases a
  simp [RoundCost.composed, RoundCost.zero]

/-- The **cost vector**: the eleven counts with the dial index forgotten, read in the field order of
the structure. Two faces can be compared on it even though their records have different indices. -/
def RoundCost.vector {σ : R} (c : RoundCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr)

/-! ## 2. The per-operation counts

### The ring layer, read off the canonical bodies -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`
(`AnyRing.lean:109-110`): five multiplications and two additions, no subtraction, no negation, no
channel read and no dial conversion, because `σ` arrives already in `R`. -/
def ringMulCost (σ : R) : RoundCost σ :=
  { mul := 5, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical channel read** `z.1 + r*z.2` (`ChannelsAnyRing.lean:139`): one multiplication
and one addition, and one application of the linear functional, counted in `channel`. -/
def ringUpperCost (σ : R) : RoundCost σ :=
  { mul := 1, add := 1, sub := 0, neg := 0, cast := 0, channel := 1,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical norm form** `N(a + bx) = a^2 - sigma*b^2` (`AnyRing.lean:128`): three
multiplications and one subtraction. The round itself performs no norm read; this count is the price
of the face-identification read of §4.2. -/
def ringNormFormCost (σ : R) : RoundCost σ :=
  { mul := 3, add := 0, sub := 1, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The additive `ZMod` layer, one `Word` addition.** The SHA-2 body adds `Word`s
(`Sha2.lean:108,109,112,116`), and a word addition modulo `2^32` is one addition of `ZMod (2^32)`
- `denote_wrapping_add` (`Denotation.lean:35`) states that correspondence for the extracted
`wrapping_add` at width 64, and it is carried below as the named premise `AddRead`, not as an axiom.
`cast` is zero because the canonical body holds its words as `Word`, not as a dial enum variant;
`wordMulCost`-style dial conversions are the extracted bodies' extra unit (AR-10 §2) and are not
part of a SHA-2 round. -/
def wordAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 1, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The two-slot reading of one addition**, `RingCell.add` on `R × R`: two scalar additions. This
is the count under decision D2 of `ops-map.md:221`, where the round's working words are held as
two-slot cells rather than as single slots. The headline round cost of §3 uses `wordAddCost`,
because the canonical body adds `Word`s; this count is stated so that the two readings are
separated by a number rather than by prose. -/
def ringCellAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The rotation reading**, `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))`: one canonical
multiplication followed by one channel read, composed. Its *count* is a fact about two bodies and is
unconditional; its *identification* with a word rotation is candidate C5 (`ops-map.md:167-180`) at
grade `model` and is carried as the named premise `RotRead` in §3.2. -/
def rotReadCost (σ : R) : RoundCost σ := ringMulCost σ + ringUpperCost σ

/-! ### The bodies the ring counts are read off -/

/-- Unfolding the canonical multiplication exhibits the four-slot expression whose five
multiplications and two additions `ringMulCost` names. -/
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
additions, one channel read, no dial conversion. -/
theorem rotReadCost_fields (σ : R) :
    rotReadCost σ =
      { mul := 6, add := 3, sub := 0, neg := 0, cast := 0, channel := 1,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } := by
  simp [rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed]

/-! ### The word layer, read off the canonical `Crypto.Sha2` bodies

Each count below is the operation inventory of the displayed body, and the display is `rfl`-checked
in this module. A count is a reading of a displayed term: nothing in Lean counts the operations of a
term, which is the residual recorded in §5 of the receipt and in AR-10 §4 item 1. -/

/-- **The rotation `ROTR`** `(x >>> (n % 32)) ||| (x <<< ((32 - n) % 32))` (`Sha2.lean:50`): two
shifts and one disjunction. The shift amounts are data, so `shift` counts both shifts and `bitOr`
counts the single `|||`; a rotation is not one operation in the canonical body. -/
def rotrCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 2, bitOr := 1 }

/-- **The logical right shift `SHR`** `x >>> (n % 32)` (`Sha2.lean:54`): one shift and nothing else. -/
def shrxCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 1, bitOr := 0 }

/-- **The message-schedule small sigma** `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`
(`Sha2.lean:60`): two rotations, one shift, two XORs. -/
def sigma0Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 5, bitOr := 2 }

/-- **The message-schedule large sigma** `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`
(`Sha2.lean:64`): the same inventory. -/
def sigma1Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 5, bitOr := 2 }

/-- **The round function `Σ₀`** `ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x` (`Sha2.lean:67`): three rotations
and two XORs, and **no shift by itself** - the shifts of FIPS 180-4 §4.1.2 belong to `σ₀` and `σ₁`
only. -/
def bigSigma0Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- **The round function `Σ₁`** `ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x` (`Sha2.lean:70`): the same
inventory as `Σ₀`. -/
def bigSigma1Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- **The choose function `Ch`** `(x &&& y) ⊕ (¬x &&& z)` (`Sha2.lean:73`): two ANDs, one NOT, two
XORs. The AND and the NOT have no ring reading at word width (O2). -/
def chooseCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 2, bitNot := 1, shift := 0, bitOr := 0 }

/-- **The majority function `Maj`** `(x &&& y) ⊕ (x &&& z) ⊕ (y &&& z)` (`Sha2.lean:76`): three ANDs
and two XORs, and no NOT. -/
def majorityCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 3, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The composition of a rotation's word-layer cost with its ring-layer reading.** One `ROTR` in the
canonical body costs two shifts and one disjunction, and under C5 it is also one `ringMul` plus one
`ringUpper`. Both counts belong to the same operation under two readings, and the record is where
the two are kept apart. -/
def rotrRotatedCost (σ : R) : RoundCost σ := rotReadCost σ + rotrCost σ

/-! ### The bodies the word counts are read off -/

/-- Unfolding `ROTR` exhibits the two shifts and the one disjunction `rotrCost` names. -/
theorem rotr_body_ops (x : Word) (n : Nat) :
    rotr x n = (x >>> (n % 32)) ||| (x <<< ((32 - n) % 32)) := rfl

/-- Unfolding `SHR` exhibits the one shift `shrxCost` names. -/
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
message schedule.** `Σ₀` and `Σ₁` are three rotations each, and a `ROTR` is two shifts, so a round's
`shift` field is exactly `2 * 3 * 2 = 12`. The `SHR^3` of `σ₀` and the `SHR^10` of `σ₁` add one shift
each, which is why a schedule step has five shifts per sigma rather than four. A round cost that
counted the two `SHR` terms would be counting a schedule step (`Sha2.lean:87-88`), where the
message schedule uses `σ₀` and `σ₁`. -/
theorem shifts_come_from_rotations (σ : R) :
    (bigSigma0Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (bigSigma1Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (sigma0Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift ∧
      (sigma1Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift := by
  unfold bigSigma0Cost bigSigma1Cost sigma0Cost sigma1Cost rotrCost shrxCost
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- The schedule step's own count, derived from the same two bodies: `σ₁` and `σ₀` and nothing
else. -/
theorem sigma0Cost_sum (σ : R) :
    sigma0Cost σ + sigma1Cost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 } := by
  simp [sigma0Cost, sigma1Cost, RoundCost.composed]

/-! ## 3. The round cost

### 3.1 The count

One round of `compress` (`Sha2.lean:100-119`) performs, per round `t`:

- `Sigma1 e` - three rotations, two XORs (`Sha2.lean:108`);
- `choose e f g` - two ANDs, one NOT, two XORs (`:108`);
- `Sigma0 a` - three rotations, two XORs (`:109`);
- `majority a b c` - three ANDs, two XORs (`:109`);
- `t1 := hh + Sigma1 e + choose e f g + K.getD t 0 + w.getD t 0` - **four** word additions
  (`:108`, left-associated, arity checked in `t1_arity` below);
- `t2 := Sigma0 a + majority a b c` - **one** word addition (`:109`);
- the state update `0 => t1 + t2` and `4 => d + t1` - **two** word additions (`:112`, `:116`).

The three projections `K.getD t 0`, `w.getD t 0` and the eight `acc.getD i 0` are element reads:
`RoundCost` has no read field, so they cost nothing here (§1). `schedule` is **not** part of a round:
it is one pass of forty-eight steps per block and is measured separately in §3.3. -/

/-- The four-addition shape of `t1`, as it is written at `Sha2.lean:108`: left-associated, four
`Word` additions and nothing else. -/
theorem t1_arity (hh s c k w : Word) :
    hh + s + c + k + w = (((hh + s) + c) + k) + w := rfl

/-- **The round cost**, at the dial `σ`. Six rotations under the C5 reading (`rotReadCost` each), the
four round-function bodies of `Crypto/Sha2.lean`, and seven word additions (`wordAddCost` each).

Temporal and spatial cost of this definition: O(1) both ways. It is a closed term - six record
compositions and four record additions, no recursion, no allocation - so **the growth is in the
counted quantity, not in the work the definition does**, the §8 "not immediate from its type" case.
Counted in: the eleven `RoundCost` fields, one per operation kind. Derivation: the per-operation
inventory of §2, composed; the recurrence it closes over is §3.2. -/
def roundCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ +
    bigSigma0Cost σ + bigSigma1Cost σ + chooseCost σ + majorityCost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    wordAddCost σ

/-- **The round's closed count.** Six multiplications-and-a-channel-read rotations, four round
functions, seven additions: 36 multiplications, 25 additions, 6 channel reads, 8 XORs, 5 ANDs, 1 NOT,
12 shifts, 6 disjunctions, and no subtraction, no negation and no dial conversion. -/
theorem roundCost_fields (σ : R) :
    roundCost σ =
      { mul := 36, add := 25, sub := 0, neg := 0, cast := 0, channel := 6,
        bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  simp [roundCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, bigSigma0Cost,
    bigSigma1Cost, chooseCost, majorityCost, wordAddCost]

/-- The round's cost vector, with the dial index forgotten. -/
theorem roundCost_vector (σ : R) : (roundCost σ).vector = (36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6) := by
  rw [roundCost_fields]
  rfl

/-- **The round's ring cost alone**, as a `RingCost`-shaped six-tuple: 36 multiplications, 25
additions, 6 channel reads. The remaining five fields are what a `RingCost σ` cannot express. -/
theorem roundCost_ring_fields (σ : R) :
    (roundCost σ).mul = 36 ∧ (roundCost σ).add = 25 ∧ (roundCost σ).sub = 0 ∧
      (roundCost σ).neg = 0 ∧ (roundCost σ).cast = 0 ∧ (roundCost σ).channel = 6 := by
  rw [roundCost_fields]
  simp

/-- **The same round under the two-slot additive reading.** 36 multiplications, 32 additions, 6
channel reads, and an unchanged non-ring layer: the seven word additions become seven `RingCell`
additions, each of which is two scalar additions, so the `add` field rises from 25 to 32. -/
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

/-- **The two-slot additive reading costs 14 more scalar additions and changes nothing else.** With
the seven word additions read as `RingCell.add` instead of as single-slot `ZMod` additions (decision
D2, `ops-map.md:221`), the round's `add` field rises from 25 to 32 and the other ten fields are the
same. The two readings are separated by a kernel-checked number rather than by prose. -/
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

/-! ### 3.2 The rotation reading, carried as a premise

The count of §3.1 is a fact about bodies and is unconditional. The *identification* of six
`ROTR`s with six `ringMul`-plus-`ringUpper` pairs is candidate C5 (`ops-map.md:167-180`) at grade
`model`, and it is carried as the named premise `RotRead`, unused by the count. -/

/-- **The rotation reading, as a named premise.** The channel read of the product
`ringMul 1 (x,0) (2^n, 2^(32-n))` is the left rotation of `x` by `n` bits, at width 32. Stated, not
proved: C5 at `ops-map.md:167-180`, grade `model`. -/
def RotRead (n : Nat) (x : W32) : Prop :=
  ringUpper faceHyperbolic (ringMul faceHyperbolic (x, 0) ((2 ^ n : Nat), (2 ^ (32 - n) : Nat))) =
    (((x.val * (2 ^ n)) % (2 ^ 32) + x.val / (2 ^ (32 - n)) : Nat) : W32)

/-- **The rotation reading is available at every face, with the same parameter.** This is AR-9's
`ringUpper_mul_eval`, re-proved here at width 32 and for every `σ` and every channel root: the read
of the product along the `x ↦ (x,0)` line is `x * (a + r*b)`, and the dial does not reach the
right-hand side because it is multiplied by `x.2 * b.2 = 0`. The premise `r ^ 2 = σ` of
`ringUpper_mul` is therefore needed for **channel multiplicativity only**, and a rotation read is not
a channel multiplication. -/
theorem channel_read_zero_slot (σ r a b x : W32) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- **The dial term of the rotation reading is present in the count and vanishes in the value.** The
multiplication `σ * z.2 * w.2` of `ringMul`'s first slot is one of the five multiplications
`ringMulCost` names, and at `z = (x,0)` its value is zero at every face. The cost pays for it
anyway: a count that omitted it would be the count of a specialised body, not of `ringMul`. -/
theorem dial_term_is_zero_valued (σ a b x : W32) :
    ringMul σ (x, 0) (a, b) = (x * a, x * b) := by
  simp only [ringMul]
  simp

/-- The rotation reading's cost is unconditional; the premise is deliberately unused. -/
theorem rotRead_cost_of_reading (σ : R) (n : Nat) (x : W32) (_h : RotRead n x) :
    (rotReadCost σ).mul = 6 ∧ (rotReadCost σ).add = 3 ∧ (rotReadCost σ).channel = 1 := by
  rw [rotReadCost_fields]
  simp

/-! ### 3.3 The schedule step, measured separately

`schedule` (`Sha2.lean:82-89`) extends the block by one step per `t = 16 … 63`, that is **48 steps per
block**, each `W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]`: two small sigmas and four word
additions. It is not part of a round. -/

/-- **One schedule-extension step's cost**: four rotations under C5, two small sigmas, four word
additions. Closed count: 24 multiplications, 16 additions, 4 channel reads, 4 XORs, 10 shifts,
4 disjunctions. -/
def schedStepCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + sigma0Cost σ + sigma1Cost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ

theorem schedStepCost_fields (σ : R) :
    schedStepCost σ =
      { mul := 24, add := 16, sub := 0, neg := 0, cast := 0, channel := 4,
        bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 } := by
  simp [schedStepCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, sigma0Cost,
    sigma1Cost, wordAddCost]

/-! ### 3.4 Composition over the 64-round chain

**Measure:** `RoundCost σ`, eleven fields, one counted quantity each (§1).

**Counted quantity:** ring operations of `R[x]/(x^2 - σ)` and word operations with no ring reading.

**Recurrence.** `roundChainCost σ 0 = RoundCost.zero σ` and
`roundChainCost σ (n + 1) = roundChainCost σ n + roundCost σ` - one round per step, plus whatever
the rounds already made have spent (`roundChainCost` below, `rfl`).

**Why it closes.** The recursion is on `Nat`, so the induction is well founded with a checked base
case; `RoundCost.rep_add` proves the recurrence telescopes, and `roundChainCost_eq` reads the
telescoping sum off: at `n` rounds every field is `n` times the corresponding field of one round and
the grouping of the `n`-fold dependent sum is irrelevant (`RoundCost.composed_assoc`,
`composed_comm`). This is AR-10's `rep_add` law, restated for the eleven fields. -/

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.** Temporal
O(1) and spatial O(1): eleven `Nat` multiplications on an eleven-field record, no recursion, no
allocation. Counted in: nothing - this is the iteration operator of the measure, not an operation of
the round; the `n` it is given is the number of repeated rounds, so the value grows while the
function does not recur. Derivation: fieldwise `n * a.field`, the componentwise image of the `n`-fold
dependent sum proved additive in `RoundCost.rep_add`. -/
def RoundCost.rep (σ : R) (n : Nat) (a : RoundCost σ) : RoundCost σ :=
  { mul := n * a.mul, add := n * a.add, sub := n * a.sub, neg := n * a.neg, cast := n * a.cast,
    channel := n * a.channel, bitXor := n * a.bitXor, bitAnd := n * a.bitAnd, bitNot := n * a.bitNot,
    shift := n * a.shift, bitOr := n * a.bitOr }

/-- **Base case: zero copies cost nothing.** -/
@[simp] theorem RoundCost.rep_zero (σ : R) (a : RoundCost σ) :
    RoundCost.rep σ 0 a = RoundCost.zero σ := by
  simp [RoundCost.rep, RoundCost.zero]

/-- **Recurrence: one more copy is one copy plus the measure of the copies already made.** -/
theorem RoundCost.rep_succ (σ : R) (n : Nat) (a : RoundCost σ) :
    RoundCost.rep σ (n + 1) a = a + RoundCost.rep σ n a := by
  cases a
  simp [RoundCost.rep, RoundCost.composed, Nat.succ_mul, Nat.add_comm]

/-- **THE COMPOSITION LAW over an iterated construction.** `n + m` copies cost the sum of `n` copies
and `m` copies, field by field. Together with `rep_succ` this is additivity of the measure over an
iterated construction stated once, for every operation kind of the measure: AR-10's
`RingCost.rep_add` restated for eleven fields. -/
theorem RoundCost.rep_add (σ : R) (n m : Nat) (a : RoundCost σ) :
    RoundCost.rep σ (n + m) a = RoundCost.rep σ n a + RoundCost.rep σ m a := by
  cases a
  simp [RoundCost.rep, RoundCost.composed, Nat.add_mul]

/-- **Iteration distributes over dependent composition**: `n` copies of two composed operations cost
`n` copies of each. This is the law that lets a link cost be stated once and iterated. -/
theorem RoundCost.rep_composed (σ : R) (n : Nat) (a b : RoundCost σ) :
    RoundCost.rep σ n (a + b) = RoundCost.rep σ n a + RoundCost.rep σ n b := by
  cases a
  cases b
  simp [RoundCost.rep, RoundCost.composed, Nat.mul_add]

/-- **The cost of a chain of `n` rounds.** Temporal `n` recursive steps, each one record addition;
spatial one accumulator and one stack frame, O(1) words beyond the result record. Counted in: the
eleven `RoundCost` fields. Derivation: the recurrence `c 0 = zero σ`, `c (n + 1) = c n + roundCost σ`,
one round per step; it closes because `RoundCost.rep_add` telescopes it to `n` copies of the round
(`roundChainCost_rep` below) and `Nat` induction has no open base case. -/
def roundChainCost (σ : R) : Nat → RoundCost σ
  | 0 => RoundCost.zero σ
  | n + 1 => roundChainCost σ n + roundCost σ

/-- **Additivity over the iterated construction, in the measure's own iteration operator.** The chain
of `n` rounds costs `n` copies of the round's cost; this is the round's counterpart of AR-10's
`chainMulCost_rep`. -/
theorem roundChainCost_rep (σ : R) (n : Nat) :
    roundChainCost σ n = RoundCost.rep σ n (roundCost σ) := by
  induction n with
  | zero => simp [roundChainCost, RoundCost.rep_zero, RoundCost.zero]
  | succ n ih => rw [roundChainCost, ih, RoundCost.rep_succ, RoundCost.composed_comm]

/-- **THE CLOSED FORM: 64 rounds.** Sixty-four rounds of `compress` are `64` copies of `roundCost`,
that is 2304 ring multiplications, 1600 ring additions, 384 channel reads, 512 XORs, 320 ANDs, 64
NOTs, 768 shifts and 384 disjunctions, and no subtraction, no negation and no dial conversion. -/
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

/-- **Additivity over the 64-round chain, the statement the wave asked for.** Two chains of `m` and
`n` rounds composed one after the other cost the sum, so the measure of the concatenation is the sum
of the measures: `RoundCost.rep_add` instantiated at the round. -/
theorem roundChainCost_add (σ : R) (m n : Nat) :
    roundChainCost σ (m + n) = roundChainCost σ m + roundChainCost σ n := by
  simp only [roundChainCost_eq]
  apply RoundCost.eq_of_fields <;> simp [RoundCost.composed, Nat.add_mul, Nat.mul_add]

/-- The chain is closed on both ends: zero rounds cost nothing and one round costs one round. -/
theorem roundChainCost_zero (σ : R) : roundChainCost σ 0 = RoundCost.zero σ := rfl

theorem roundChainCost_one (σ : R) : roundChainCost σ 1 = roundCost σ := by
  simp [roundChainCost, RoundCost.zero, RoundCost.composed]

/-- The measure is not degenerate on the iterated construction: a longer chain is never free in the
multiplication field. -/
theorem roundChainCost_strict (σ : R) {m n : Nat} (h : m < n) :
    (roundChainCost σ m).mul < (roundChainCost σ n).mul := by
  simp only [roundChainCost_eq]
  omega

/-! ### 3.5 Per 64-byte block

One `compress` is one §6.2.2 block transformation: sixty-four rounds, **one** `schedule` pass of
forty-eight steps (`Sha2.lean:97`), and the eight-word add-back `h.getD i 0 + v.getD i.val 0`
(`Sha2.lean:120`). The block cost is the sum of those three, and it is the figure a per-block number
would be; §5 records that this module does not convert it into a per-64-byte-block figure for an
arbitrary message length. -/

/-- **The cost of one §6.2.2 block transformation**: the 64-round chain, forty-eight schedule steps,
and eight add-back additions. Temporal and spatial O(1): a closed term, no recursion. Counted in: the
eleven `RoundCost` fields. Derivation: the three components above, composed; closure by
`RoundCost.rep_add` and `roundChainCost_eq`. -/
def blockCost (σ : R) : RoundCost σ :=
  roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ) + RoundCost.rep σ 8 (wordAddCost σ)

/-- **The block cost is the 64-round cost plus exactly the schedule and the add-back.** The split is
stated so that a reader can see which part of the block figure is rounds and which is not. -/
theorem blockCost_split (σ : R) :
    blockCost σ =
      roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ) + RoundCost.rep σ 8 (wordAddCost σ) :=
  rfl

/-- **The block's closed count.** 3456 ring multiplications, 2376 ring additions, 576 channel reads,
704 XORs, 320 ANDs, 64 NOTs, 1248 shifts and 576 disjunctions, and no subtraction, no negation and
no dial conversion. -/
theorem blockCost_eq (σ : R) :
    blockCost σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576 } := by
  rw [blockCost_split, roundChain64, schedStepCost_fields]
  rfl

/-! ## 4. The three faces

### 4.1 What is computed, and what the answer is

The round cost is computed at all three dial values, `σ = -1` (elliptic), `σ = 0` (parabolic) and
`σ = 1` (hyperbolic), and **the three computations agree in all eleven fields.**

The reason is a property of the counted bodies, not a property of the algebra: every count in §2 is a
closed record of `Nat` literals, and no counted body - `ringMul`, `ringUpper`, `ringNormForm`,
`RingCell.add`, `rotr`, `shrx`, `sigma0`, `sigma1`, `Sigma0`, `Sigma1`, `choose`, `majority` or
`BitVec` addition - has a dial parameter or a branch on one. `ringMul` takes `σ` as an argument and
its body is one expression for every `σ` (`ringMul_body_ops`); `Word` operations do not mention a
dial at all. So the measure is **σ-blind by construction**, and the honest form of the result is a
statement about the counted bodies. It is **not** a statement that the trichotomy does not affect
cost: §4.2 and §4.3 name the operations whose counting would make a face difference appear, and one
of them makes the elliptic face's figure *undefined* rather than different.

### 4.2 Where the σ actually is: the dial-reading operations

Three operations read the dial. They are counted separately from the round's own counts, and each is
priced at every face.

| dial-reading operation | count | face dependence |
|---|---|---|
| channel root, the parameter `r` of `ringUpper r z` | one argument, **0** operations | two faces admit a constant root, the third admits none (§4.4) |
| channel multiplicativity `ringUpper_mul`, the consumer of `r ^ 2 = σ` | `ringUpperCost`, one per use | two faces admissible, one **not** |
| face certification, a norm-form read whose residue names the face | `ringNormFormCost`, ⟨3,0,1⟩ | identical on all three; its *outcome* differs (§4.3) |
| dial conversion, the extracted `from_i64` of `cell.Sigma` | one `cast` per two-slot operation | identical on all three; absent from the canonical body |

The rotation reading itself is **not** a dial-reading operation: `channel_read_zero_slot` evaluates
it for every `σ` and every `r`, because the dial term of `ringMul` is annihilated by the zero second
slot of `(x, 0)`. The premise `r ^ 2 = σ` is needed only for `ringUpper_mul`'s channel
multiplicativity, which a rotation read does not use (AR-9's `ringUpper_mul_eval`). -/

/-- **The round cost does not depend on the dial.** For every `σ` at width 32 the eleven counts are
`36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6`. -/
theorem roundCost_faceBlind (σ : W32) :
    (roundCost σ).vector = (36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6) :=
  roundCost_vector σ

/-- **THE THREE FACES, COMPUTED AND COMPARED.** Elliptic `-1`, parabolic `0` and hyperbolic `1` give
the same eleven counts, field by field. -/
theorem roundCost_threeFaces :
    (roundCost faceElliptic).vector = (36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6) ∧
      (roundCost faceParabolic).vector = (36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6) ∧
      (roundCost faceHyperbolic).vector = (36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6) :=
  ⟨roundCost_faceBlind _, roundCost_faceBlind _, roundCost_faceBlind _⟩

/-- **The three faces, field by field, all eleven counts.** The explicit statement, so the comparison
does not rest on the tuple encoding: no field differs at any face. -/
theorem roundCost_threeFaces_fieldwise :
    (roundCost faceElliptic).mul = (roundCost faceParabolic).mul ∧
      (roundCost faceElliptic).mul = (roundCost faceHyperbolic).mul ∧
      (roundCost faceElliptic).add = (roundCost faceParabolic).add ∧
      (roundCost faceElliptic).add = (roundCost faceHyperbolic).add ∧
      (roundCost faceElliptic).sub = (roundCost faceParabolic).sub ∧
      (roundCost faceElliptic).sub = (roundCost faceHyperbolic).sub ∧
      (roundCost faceElliptic).neg = (roundCost faceParabolic).neg ∧
      (roundCost faceElliptic).neg = (roundCost faceHyperbolic).neg ∧
      (roundCost faceElliptic).cast = (roundCost faceParabolic).cast ∧
      (roundCost faceElliptic).cast = (roundCost faceHyperbolic).cast ∧
      (roundCost faceElliptic).channel = (roundCost faceParabolic).channel ∧
      (roundCost faceElliptic).channel = (roundCost faceHyperbolic).channel ∧
      (roundCost faceElliptic).bitXor = (roundCost faceParabolic).bitXor ∧
      (roundCost faceElliptic).bitXor = (roundCost faceHyperbolic).bitXor ∧
      (roundCost faceElliptic).bitAnd = (roundCost faceParabolic).bitAnd ∧
      (roundCost faceElliptic).bitAnd = (roundCost faceHyperbolic).bitAnd ∧
      (roundCost faceElliptic).bitNot = (roundCost faceParabolic).bitNot ∧
      (roundCost faceElliptic).bitNot = (roundCost faceHyperbolic).bitNot ∧
      (roundCost faceElliptic).shift = (roundCost faceParabolic).shift ∧
      (roundCost faceElliptic).shift = (roundCost faceHyperbolic).shift ∧
      (roundCost faceElliptic).bitOr = (roundCost faceParabolic).bitOr ∧
      (roundCost faceElliptic).bitOr = (roundCost faceHyperbolic).bitOr := by
  rw [roundCost_fields faceElliptic, roundCost_fields faceParabolic,
    roundCost_fields faceHyperbolic]
  decide

/-- **The 64-round chain, all three faces.** The chain inherits the round's σ-blindness because it
is `n` copies of it. -/
theorem roundChainCost_threeFaces :
    (roundChainCost faceElliptic 64).vector = (roundChainCost faceParabolic 64).vector ∧
      (roundChainCost faceParabolic 64).vector = (roundChainCost faceHyperbolic 64).vector := by
  simp [roundChainCost_eq, RoundCost.vector, RoundCost.rep]

/-- **The block cost, all three faces.** -/
theorem blockCost_threeFaces :
    (blockCost faceElliptic).vector = (blockCost faceParabolic).vector ∧
      (blockCost faceParabolic).vector = (blockCost faceHyperbolic).vector := by
  simp [blockCost_eq, RoundCost.vector]

/-! ### 4.3 The face is visible in the algebra, through the norm image -/

/-- The parabolic norm image reduced mod 8, `{0, 1, 4}` (`AlgebraNorm.lean`, AR-7). -/
def resParabolic : Finset (ZMod 8) := {0, 1, 4}

/-- The elliptic norm image reduced mod 8, `{0, 1, 2, 4, 5}`. -/
def resElliptic : Finset (ZMod 8) := {0, 1, 2, 4, 5}

/-- The hyperbolic norm image reduced mod 8, `{0, 1, 3, 4, 5, 7}`. -/
def resHyperbolic : Finset (ZMod 8) := {0, 1, 3, 4, 5, 7}

/-- **The three mod-8 norm images, one kernel decision each over all 64 points.** These are the
face-distinguishing certificates of AR-7, re-proved here so that §4 rests on this module's own rows
rather than on a citation. -/
theorem image_mod8_elliptic :
    ∀ z : ZMod 8 × ZMod 8, ringNormForm (-1 : ZMod 8) z ∈ resElliptic := by decide
theorem image_mod8_parabolic :
    ∀ z : ZMod 8 × ZMod 8, ringNormForm (0 : ZMod 8) z ∈ resParabolic := by decide
theorem image_mod8_hyperbolic :
    ∀ z : ZMod 8 × ZMod 8, ringNormForm (1 : ZMod 8) z ∈ resHyperbolic := by decide

/-- **The three norm images are pairwise different.** The faces are separable by the image of the
norm form, so a face label is available to the cost record without touching the count. -/
theorem normImages_pairwise_distinct :
    resElliptic ≠ resParabolic ∧ resParabolic ≠ resHyperbolic ∧ resElliptic ≠ resHyperbolic := by
  decide

/-- **A residue that names the elliptic face and no other.** `N ≡ 2 (mod 8)` is attained at the
elliptic face and at neither of the other two, so one norm-form read is enough to identify the face
that a two-slot computation is running at. This is the dial-reading operation that a face-dependent
cost record would have to count: `ringNormFormCost`, `⟨3, 0, 1⟩`, the same at all three faces, with
an outcome that differs. -/
theorem normImage2_names_elliptic :
    (∃ z : ZMod 8 × ZMod 8, ringNormForm (-1 : ZMod 8) z = 2) ∧
      (∀ z : ZMod 8 × ZMod 8, ringNormForm (0 : ZMod 8) z ≠ 2) ∧
      (∀ z : ZMod 8 × ZMod 8, ringNormForm (1 : ZMod 8) z ≠ 2) := by decide

/-- **The price of face certification.** One norm read (`ringNormFormCost`) and one channel read
(`ringUpperCost`) on top of a round. -/
theorem faceCertifyCost (σ : W32) :
    roundCost σ + ringNormFormCost σ + ringUpperCost σ =
      { mul := 40, add := 26, sub := 1, neg := 0, cast := 0, channel := 7,
        bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  rw [roundCost_fields]
  rfl

/-- **Face certification costs the same on all three faces.** The eleven counts are 40 multiplications,
26 additions, 1 subtraction, 7 channel reads and an unchanged non-ring layer at the elliptic, the
parabolic and the hyperbolic face alike: the certification read is one norm form and one channel
read, and both are the same count at every face. -/
theorem faceCertifyCost_threeFaces :
    (roundCost faceElliptic + ringNormFormCost faceElliptic + ringUpperCost faceElliptic).vector =
        (roundCost faceParabolic + ringNormFormCost faceParabolic + ringUpperCost faceParabolic).vector ∧
      (roundCost faceParabolic + ringNormFormCost faceParabolic + ringUpperCost faceParabolic).vector =
        (roundCost faceHyperbolic + ringNormFormCost faceHyperbolic
          + ringUpperCost faceHyperbolic).vector := by
  rw [faceCertifyCost, faceCertifyCost, faceCertifyCost]
  exact ⟨rfl, rfl⟩

/-! ### 4.4 The one face where a dial read is impossible

`ringUpper r z` is defined for every `r`, and `ringUpper_mul`'s premise `r ^ 2 = σ` is satisfiable at
the hyperbolic face (`r = 1`) and at the parabolic face (`r = 0`). At the elliptic face at width 32 it
is not: no `r` in `ZMod (2 ^ 32)` satisfies `r ^ 2 = -1`. The round's own figures do not use that
premise, so the elliptic round cost is a finite record like the other two; what fails at the elliptic
face is **channel multiplicativity**, and with it any reading that pushes a channel read through a
product of two general two-slot elements. That is the difference between the faces: not a different
number, but a *refused* one. -/

/-- **No root of `-1` in `ZMod 4`**, decided over the four elements. -/
theorem no_root_neg_one_w4 : ∀ r : ZMod 4, r ^ 2 ≠ (-1 : ZMod 4) := by decide

/-- **No root of `-1` at the SHA-2 word width `ZMod (2 ^ 32)`.** A root at width 32 would reduce, by
the ring hom `ZMod (2 ^ 32) →+* ZMod 4` along `4 ∣ 2 ^ 32`, to a root at width 4, which the previous
theorem excludes. -/
theorem no_root_neg_one_w32 : ∀ r : W32, r ^ 2 ≠ (-1 : W32) := by
  intro r h
  have h' := congrArg (ZMod.castHom (n := 2 ^ 32) (m := 4) (by decide : (4 : ℕ) ∣ 2 ^ 32)
    (ZMod 4)) h
  rw [map_pow, map_neg, map_one] at h'
  exact no_root_neg_one_w4 _ h'

/-- **The two faces that admit a constant root.** The premise of `ringUpper_mul` is an arithmetic
identity at each of them, discharged by `rfl`, not a number-theory hypothesis. -/
theorem root_hyperbolic : (1 : W32) ^ 2 = faceHyperbolic := rfl

theorem root_parabolic : (0 : W32) ^ 2 = faceParabolic := rfl

/-- **Channel multiplicativity is admissible at two faces and refused at the third.** This is the
sharpest per-face statement in the module, and it is about *admissibility of a reading*, not about a
count: `ringUpper_mul` is `proved` at `σ = 1` and at `σ = 0`, and at `σ = -1` no instantiation of its
premise exists at this width. -/
theorem channelMul_admissible_faces :
    (ringUpper faceHyperbolic (ringMul faceHyperbolic (0, 1) (0, 1)) =
        ringUpper faceHyperbolic (0, 1) * ringUpper faceHyperbolic (0, 1)) ∧
      (ringUpper faceParabolic (ringMul faceParabolic (0, 1) (0, 1)) =
        ringUpper faceParabolic (0, 1) * ringUpper faceParabolic (0, 1)) ∧
      (∀ r : W32, r ^ 2 ≠ faceElliptic) :=
  ⟨ringUpper_mul (r := faceHyperbolic) (by rfl) _ _, ringUpper_mul (r := faceParabolic) (by rfl) _ _,
    fun r => no_root_neg_one_w32 r⟩

/-- **The rotation reading needs no root at all.** A round's six rotations are channel reads of
products along the `x ↦ (x, 0)` line, and the dial term of `ringMul` is multiplied by `0 * 0` there,
so the reading is available at all three faces with the same parameter. This is why the elliptic face
has a finite round cost even though it has no channel root. -/
theorem rotationReading_needs_no_root (σ r : W32) :
    ∀ a b x : W32, ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) :=
  channel_read_zero_slot σ r

/-! ## 5. What this module does not give

No bound in time, latency, bytes, energy, dispatches or hardware, and no asymptotic statement
(`AGENTS.md:429`). No per-byte-block figure for a message of a given length: `blockCost` is the cost
of **one** §6.2.2 block transformation and §6 of the receipt names the conversion as residual. No
element-traffic count: `RoundCost` has no read field, so the projections of the round body cost
nothing here. No cost for the padding, the loading loop or the digest store. The residual list is in
`work/proofs/AlgebraCostRound.md`. -/

/-! ## 6. Audit rows

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
#print axioms sigma0Cost_sum
#print axioms t1_arity
#print axioms roundCost_fields
#print axioms roundCost_vector
#print axioms roundCost_ring_fields
#print axioms roundCostTwoSlot_fields
#print axioms roundCost_twoSlot_add_only
#print axioms channel_read_zero_slot
#print axioms dial_term_is_zero_valued
#print axioms rotRead_cost_of_reading
#print axioms schedStepCost_fields
#print axioms roundChainCost_rep
#print axioms roundChainCost_eq
#print axioms roundChain64
#print axioms roundChainCost_add
#print axioms roundChainCost_zero
#print axioms roundChainCost_one
#print axioms roundChainCost_strict
#print axioms blockCost_eq
#print axioms blockCost_split
#print axioms roundCost_faceBlind
#print axioms roundCost_threeFaces
#print axioms roundCost_threeFaces_fieldwise
#print axioms roundChainCost_threeFaces
#print axioms blockCost_threeFaces
#print axioms image_mod8_elliptic
#print axioms image_mod8_parabolic
#print axioms image_mod8_hyperbolic
#print axioms normImages_pairwise_distinct
#print axioms normImage2_names_elliptic
#print axioms faceCertifyCost
#print axioms faceCertifyCost_threeFaces
#print axioms no_root_neg_one_w4
#print axioms no_root_neg_one_w32
#print axioms root_hyperbolic
#print axioms root_parabolic
#print axioms channelMul_admissible_faces
#print axioms rotationReading_needs_no_root

end algebra_cost_round