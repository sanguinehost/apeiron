/-
SCOPE: work/proofs/AlgebraCost.lean, new, and its receipt work/proofs/AlgebraCost.md. The
smallest kernel-checked cost core for the two-slot algebra `R[x]/(x^2 - sigma)` over the ring the
extracted word kernels run in: one named measure of ring operations, the operation count of the
canonical definitions `ringMul`, `ringNormForm`, `ringUpper`, `ringLower`, the counts of the
generated bodies `word_mul` and `word_norm` anchored on their proved denotations, the iteration
operator `RingCost.rep` and its additivity law, the measure of the canonical iterated construction
`ringFold` and of a length-indexed chain of links, and the cost of two crypto-relevant operations
read off `work/algebra-rooting/ops-map.md`. Complete when this module kernel-checks with every
headline audited, a seeded per-body count defect rejects, and the receipt records the counts, the
grades and the named premises.
REGISTER: algebra/resource model. No production, pin, API, extraction-root, corpus or gate-driver
edit; no Rust source, no `cargo`, no `regen.sh`, no measurement. Excludes latency, bandwidth,
cache behaviour, rustc/LLVM codegen, the kernel ABI, device dispatch cost, constant-time or
security properties of any hash, and every claim about hardware timing.
CORPUS: `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringConj,ringNormForm,ringOne,ringMul_assoc,
ringMul_comm,ringNormForm_mul}` at `Trichotomy/AnyRing.lean:98,109,118,128,146,191,180,168` and
`Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper:139,ringLower:149,ringFold:498,ringUpper_mul:160,
ringUpper_fold:509,ringLower_fold:524,ringNormForm_fold:547}` at `Trichotomy/ChannelsAnyRing.lean`,
read at the pinned revision `880aa57e47e782ed7a9d9ee1974cfc87360b446b` through
`repos/apeiron/proof/corpus/.lake/packages/apeiron/proof/`, grade `proved` in the pinned corpus
(`Apeiron/Audit.lean:1201` audits `AnyRing.ringUpper`; `:1203,1227,1236` audit `ringUpper_mul`,
`ringUpper_fold`, `ringUpper_fold_bool`). `proof/corpus/WordRefinement.lean:38,45,56`
(`denoteSigma`, `word_mul_denote`, `word_norm_denote`) and `proof/Denotation.lean:29,35`
(`denote_from_i64`, `denote_wrapping_add`) are the extracted-side links, grade `proved` by corpus
build inclusion. No `Cost`, `Steps`, `Budget` or `Measure` over ring operations exists in that
corpus; see `work/cost-model/inventory.md`.
REFERENCE: the unpublished resource model/CostAlgebra.lean:34-45 (one
field per resource, a field never mixes two resources), :80 `costAdd` (dependent composition),
:93 `overlap`, :255 `registeredCost` (the scalar an admission certificate compares),
:271 `scalar_not_sufficient` (why this module keeps the record),
:308-383 `seq`/`layerCost`/`planCost`; the unpublished resource model/
CostBridge.lean:48-65 (`structure Perf`, the source-level count record whose fields are body visits,
checked multiply-adds, reads, writes, bytes and allocations) and :88-103 (`toCost`, the map into the
registered record, with its per-visit constants graded `model`); :161 `loads_invisible_to_order`;
RowDotResource.lean:368-411 (`RowTrace`, the finite certificate of an extracted body, and
`row_trace_count`); RowDotAdmission.lean:683-729 (`srcOps`/`srcReads`, declared per-body constants)
and :903 (`improves_entry`, the strict `registeredCost` improvement an admission certificate must
show). The corpus's own cost recurrence, as the precedent for §3: `Nous/Combinator/Sphere.lean:91`
`fwhtCost` with its closed form at :100.
GATE: lane lake env lean
~/icarus/work/proofs/AlgebraCost.lean, cwd
proof/corpus (absolute `bin/lane` from the workspace
root), which resolves the pinned `apeiron` and `mathlib` packages. Observed red before this file
existed: `lake env lean` on the absent path exits 1 (`error: file does not exist`). Wall-clock
budget 590 s for the invocation, the seed control and the restore, under
`flock /tmp/apeiron/apeiron-lane.lock`, `MemoryMax=3800M`, `MemoryHigh=3400M`,
`MemorySwapMax=0`, `LEAN_NUM_THREADS=1`.
GRADE: `proved` for the measure's laws, for every operation count that is read off a definition
body by unfolding it in this module, for the composition of the measure over an iterated
construction, and for the two worked numeric examples; the closed cone is
`{propext, Classical.choice, Quot.sound}` with no `sorryAx`, checked by the `#print axioms` rows at
the end of this file and recorded in the receipt. `model` for the two facts that are read off
*generated* body text rather than off a Lean definition body in this module: the
`scalarRingInst.mul`/`add`/`sub`/`from_i64` inventory of `cell.Cell.mul` and `cell.Cell.norm_form`,
graded `model` exactly as `CostBridge`'s `srcOps`/`srcReads` per-visit constants are graded
(`CostBridge.lean:28-31`, `RowDotAdmission.lean:43-45`). `conditional`: the identification of
`ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(w-n)))` with a word rotation, which is the candidate `C5`
of `work/algebra-rooting/ops-map.md:167-180` at grade `model` and is carried below as the named
premise `RotRead` of `rotation_cost`, never as an axiom. Unmeasured and unclaimed: everything on
the REGISTER exclusion list.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
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

namespace algebra_cost

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open apeiron_word

universe u

variable {R : Type u} [CommRing R]

/-! ## The two rings the module speaks in

The ring the extracted `Cell<i64>` arithmetic runs in is `ZMod (2 ^ 64)`, fixed by
`WordRefinement.lean:38`; a crypto kernel names its word width `w` and runs in `ZMod (2 ^ w)`, and
`w = 64` recovers the extracted ring. The extracted bodies exist at width 64 only, so every count
that concerns `word_mul` or `word_norm` is stated at that width. -/

-- The ring the extracted `Cell<i64>` arithmetic runs in is `ZMod (2 ^ 64)`
-- (`WordRefinement.lean:38`), and a crypto kernel names its word width `w` and runs in
-- `ZMod (2 ^ w)`. Both are written out rather than abbreviated: an abbreviation carrying a
-- universe annotation is what the first draft did, and `RingCost` is indexed by its ring, so the
-- index universes must match exactly. `w = 64` recovers the extracted ring.

/-! ## 1. The measure

One record, one counted quantity per field, no field mixing two quantities: the convention
`CostAlgebra.lean:34-45` states for its seven registered resources, applied here to the ring
operations. Every field counts one thing and nothing else.

### Why these six quantities

The choice is forced by the corpus's own counted quantities, not chosen for convenience.

- **The tree's only formal per-primitive count records separate exactly these kinds of thing.**
  `CostBridge.Perf` (`CostBridge.lean:58`) has separate fields for `mulAdds` and for element
  traffic, and proves at `:161 loads_invisible_to_order` that the registered order is *blind* to
  the traffic because `toCost` (`:102`) maps it to a zero field. A measure that follows the
  `CostAlgebra` convention must therefore give a different field to every quantity it wants to be
  able to see. Six fields is the count needed to keep `ringMul`, `ringNormForm`, `ringUpper` and
  `ringLower` distinguishable, which is the whole point of counting ring operations rather than
  reusing `CostAlgebra.ops`.
- **The operations that appear in the bodies of this algebra are exactly multiplication,
  addition, subtraction and negation.** `ringMul` (`:109`) is five multiplications and two
  additions; `ringNormForm` (`:128`) is multiplications and one subtraction; `ringConj` (`:118`)
  is one negation. A fourth arithmetic kind does not occur in these bodies, so a fourth field
  would be unused.
- **`channel` is separate because the corpus separates it.** `ChannelsAnyRing.lean:41` tabulates
  the channel theorems as statements *about* `AnyRing.ringMul`, and `ringUpper` (`:139`) is not a
  ring operation on a cell at all: it applies a root-parameterised linear functional and returns
  an `R`. Counting it as an `add` would merge a coordinate read with a sum, which is the same
  mistake `CostAlgebra.lean:34-45` forbids.
- **`cast` is separate because the generated bodies pay for it and the canonical ones do not.**
  `cell.Cell.mul` converts its `Sigma` enum variant into the scalar ring (`ApeironWord.lean:133`,
  one `from_i64`), while `ringMul` receives `σ` already in `R` and converts nothing. Folding that
  operation into `mul` would make the correspondence between the extracted and canonical counts
  inexact by one operation, and the inexactness is the interesting part.

### What the measure deliberately excludes, and why

**Carries.** A carry is not counted, because in the ring this module measures in there is no
separate carry operation: `denote_wrapping_add` (`Denotation.lean:35`) states that the extracted
`i64` addition *is* the `ZMod` addition `denote x + denote y`, so the carry has already been
absorbed by the ring's own arithmetic before any operation is counted. A carry count would belong
to a word-level measure counted in bit operations, not to a ring measure; `work/proofs/
AlgebraCarry.lean` of this wave is where that measure would live. **Memory, allocation, dispatch
and peak live bytes** are excluded because `CostAlgebra.Cost` already owns those quantities and a
second record would duplicate them under different names. **Time and every asymptotic claim** are
excluded by `AGENTS.md:422`: a cost-model proof is not a hardware speed measurement. -/

/-- `RingCost σ` is the measure: the number of scalar ring operations one two-slot computation
performs in the ring `R` at dial value `σ`. The counted quantities are the operations of the
*algebra* `R[x]/(x^2 - σ)`, not the operations of a host, so the record carries no timing, no
byte count and no allocation.

- `mul`: scalar ring multiplications.
- `add`: scalar ring additions.
- `sub`: scalar ring subtractions.
- `neg`: scalar ring negations.
- `cast`: conversions of a dial value into the ring, that is applications of the scalar ring's
  `from` operation.
- `channel`: channel reads, that is applications of a root-parameterised linear functional to a
  two-slot element.

Dependence on `σ` is the only dependence recorded: `σ` enters the operation count of a chain of
multiplications in §3, and it is what the extracted bodies pay one `from` application for. -/
structure RingCost (σ : R) where
  mul : Nat
  add : Nat
  sub : Nat
  neg : Nat
  cast : Nat
  channel : Nat
  deriving DecidableEq, Repr

/-- The record of a computation that performs no ring operation. -/
def RingCost.zero (σ : R) : RingCost σ := ⟨0, 0, 0, 0, 0, 0⟩

/-- **Dependent composition.** A computation that runs after another performs both sets of
operations, so every count totals: this is `CostAlgebra.costAdd` (`:80`) read over ring
operations, and it is the only composition the module uses. -/
def RingCost.composed (σ : R) (a b : RingCost σ) : RingCost σ :=
  ⟨a.mul + b.mul, a.add + b.add, a.sub + b.sub, a.neg + b.neg, a.cast + b.cast,
    a.channel + b.channel⟩

instance instAddRingCost (σ : R) : Add (RingCost σ) := ⟨RingCost.composed σ⟩

/-- The dependent sum, unfolded, so composition rewrites and its laws close. -/
@[simp] theorem RingCost.composed_apply (σ : R) (a b : RingCost σ) :
    a + b = RingCost.composed σ a b := rfl

/-- **Equality of counts is fieldwise equality.** A record with one field per quantity has no room
for a hidden quantity, so two records are equal exactly when every count is equal. -/
theorem RingCost.eq_of_fields {σ : R} {a b : RingCost σ} (h1 : a.mul = b.mul) (h2 : a.add = b.add)
    (h3 : a.sub = b.sub) (h4 : a.neg = b.neg) (h5 : a.cast = b.cast) (h6 : a.channel = b.channel) :
    a = b := by
  cases a
  cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6
  rfl

/-- **Composition is commutative**: disjoint work sums the same whatever order it is listed in. -/
theorem RingCost.composed_comm (σ : R) (a b : RingCost σ) : a + b = b + a := by
  cases a
  cases b
  simp [RingCost.composed, Nat.add_comm]

/-- **Composition is associative**: the grouping of a dependent chain does not change its cost,
which is what lets a chain be assembled left- or right-associated. -/
theorem RingCost.composed_assoc (σ : R) (a b c : RingCost σ) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [RingCost.composed, Nat.add_assoc]

/-- **Composition with the zero record is neutral.** -/
theorem RingCost.composed_zero (σ : R) (a : RingCost σ) : a + RingCost.zero σ = a := by
  cases a
  simp [RingCost.composed, RingCost.zero]

/-- **Componentwise order.** The measure is compared by count, never by a weighted sum: a
single scalar hides a regression in one quantity (`CostAlgebra.scalar_not_sufficient`,
`CostAlgebra.lean:271`). -/
def RingCost.le {σ : R} (a b : RingCost σ) : Prop :=
  a.mul ≤ b.mul ∧ a.add ≤ b.add ∧ a.sub ≤ b.sub ∧ a.neg ≤ b.neg ∧ a.cast ≤ b.cast ∧
    a.channel ≤ b.channel

instance instLERingCost (σ : R) : LE (RingCost σ) := ⟨RingCost.le⟩

/-- Fieldwise order introduction. -/
theorem RingCost.le_of_fields {σ : R} {a b : RingCost σ} (h1 : a.mul ≤ b.mul) (h2 : a.add ≤ b.add)
    (h3 : a.sub ≤ b.sub) (h4 : a.neg ≤ b.neg) (h5 : a.cast ≤ b.cast) (h6 : a.channel ≤ b.channel) :
    a ≤ b :=
  ⟨h1, h2, h3, h4, h5, h6⟩

/-- The order is reflexive on every record. -/
theorem RingCost.le_refl' {σ : R} (a : RingCost σ) : a ≤ a :=
  ⟨Nat.le_refl _, Nat.le_refl _, Nat.le_refl _, Nat.le_refl _, Nat.le_refl _, Nat.le_refl _⟩

/-- The order is transitive. -/
theorem RingCost.le_trans' {σ : R} {a b c : RingCost σ} (h₁ : a ≤ b) (h₂ : b ≤ c) : a ≤ c :=
  ⟨Nat.le_trans h₁.1 h₂.1, Nat.le_trans h₁.2.1 h₂.2.1, Nat.le_trans h₁.2.2.1 h₂.2.2.1,
    Nat.le_trans h₁.2.2.2.1 h₂.2.2.2.1, Nat.le_trans h₁.2.2.2.2.1 h₂.2.2.2.2.1,
    Nat.le_trans h₁.2.2.2.2.2 h₂.2.2.2.2.2⟩

/-- The order is antisymmetric. -/
theorem RingCost.le_antisymm' {σ : R} {a b : RingCost σ} (h₁ : a ≤ b) (h₂ : b ≤ a) : a = b :=
  RingCost.eq_of_fields (Nat.le_antisymm h₁.1 h₂.1) (Nat.le_antisymm h₁.2.1 h₂.2.1)
    (Nat.le_antisymm h₁.2.2.1 h₂.2.2.1) (Nat.le_antisymm h₁.2.2.2.1 h₂.2.2.2.1)
    (Nat.le_antisymm h₁.2.2.2.2.1 h₂.2.2.2.2.1) (Nat.le_antisymm h₁.2.2.2.2.2 h₂.2.2.2.2.2)

instance instPartialOrderRingCost (σ : R) : PartialOrder (RingCost σ) where
  le := RingCost.le
  le_refl := fun a => RingCost.le_refl' (σ := σ) a
  le_trans := fun _ _ _ h₁ h₂ => RingCost.le_trans' (σ := σ) h₁ h₂
  le_antisymm := fun _ _ h₁ h₂ => RingCost.le_antisymm' (σ := σ) h₁ h₂

/-! ## 2. The operation counts of the canonical definitions

Every count below is a closed term of `Nat`, read off the body of the definition by unfolding it in
this module and naming the applications. The reading itself is one unfold per definition, O(1) in
the size of the definition, and the definitions here are one expression each.

The dial conversion is counted once per two-slot operation: `ringMul` and `ringNormForm` each read
their `σ` out of the scalar ring, which is one `from` application in the generated body. The
canonical definitions take `σ : R` already in the ring and convert nothing, and no `ringUpper`
converts anything because a root is passed in. -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`
(`AnyRing.lean:109-110`). Its body is one expression: five multiplications `z.1*w.1`, `sigma*z.2`,
`(sigma*z.2)*w.2`, `z.1*w.2`, `z.2*w.1` and two additions, one per slot. No subtraction, no
negation, no channel read, and no dial conversion, because `σ` arrives already in `R`. -/
def ringMulCost (σ : R) : RingCost σ := ⟨5, 2, 0, 0, 0, 0⟩

/-- **The canonical norm form** `N(a + bx) = a^2 - sigma*b^2` (`AnyRing.lean:128`). Its body is
`z.1 ^ 2 - σ * z.2 ^ 2`, that is `z.1*z.1 - σ*(z.2*z.2)` once the squares are opened: three
multiplications, one subtraction, and no dial conversion, because `σ` arrives already in `R`. -/
def ringNormFormCost (σ : R) : RingCost σ := ⟨3, 0, 1, 0, 0, 0⟩

/-- **The canonical channel read** `z.1 + r*z.2` (`ChannelsAnyRing.lean:139`). Its body is one
multiplication and one addition, and it converts no dial because the root `r` is an argument. -/
def ringUpperCost (σ : R) : RingCost σ := ⟨1, 1, 0, 0, 0, 1⟩

/-- **The canonical second-coordinate read** `z.1 - r*z.2` (`ChannelsAnyRing.lean:149`). Its body is
one multiplication and one subtraction, and it converts no dial because the root `r` is an
argument. It is counted beside `ringUpperCost` because the corpus's own statement of a fold's size
reads *both* channels per element (`ringNormForm_fold`, `ChannelsAnyRing.lean:547`). -/
def ringLowerCost (σ : R) : RingCost σ := ⟨1, 0, 1, 0, 0, 1⟩

/-- Unfolding the second channel read exhibits the one multiplication and one subtraction its count
names. -/
theorem ringLower_body_ops (r : R) (z : RingCell R) : ringLower r z = z.1 - r * z.2 := rfl

/-- **The canonical conjugation** `(a, -b)` (`AnyRing.lean:118`). Its body negates the second slot:
one negation, nothing else. -/
def ringConjCost (σ : R) : RingCost σ := ⟨0, 0, 0, 1, 0, 0⟩

/-- **The two-slot addition** `RingCell.add`, componentwise addition on the two slots. Its body is
two additions and nothing else; `RingCell R` is `R × R` with the componentwise structure, so this
count is read off the structure, not off a canonical `abbrev`. -/
def ringAddCost (σ : R) : RingCost σ := ⟨0, 2, 0, 0, 0, 0⟩

/-- **The extracted wrapping addition** `core.num.I64.wrapping_add`. The denotation
`denote_wrapping_add` (`Denotation.lean:35-38`) says this single extracted operation denotes the
ring addition of two words, so its count is one ring addition at width 64 and nothing else. -/
def wrappingAddCost (σ : R) : RingCost σ := ⟨0, 1, 0, 0, 0, 0⟩

/-- **Cost of a rotation under the two-slot reading**, `ringUpper r (ringMul σ z w)`: one
multiplication and one channel read, composed. This is the cost the candidate of
`work/algebra-rooting/ops-map.md:167-180` assigns to one word rotation. -/
def rotationCost (σ : R) : RingCost σ := ringMulCost σ + ringUpperCost σ

/-- The rotation's count, read off the two canonical bodies: six multiplications, three additions,
one channel read, and no dial conversion on the canonical side. -/
theorem rotationCost_fields (σ : R) : rotationCost σ = ⟨6, 3, 0, 0, 0, 1⟩ := by
  simp [rotationCost, ringMulCost, ringUpperCost, RingCost.composed]

/-! ### The bodies the counts are read off

The statements below are the module's link between a count and a body. Each one unfolds a canonical
definition in this module and exhibits the expression whose operations the count names, so a reader
can check a count against its body without reading generated text. -/

/-- Unfolding the canonical multiplication exhibits the four-slot expression whose five
multiplications and two additions `ringMulCost` names. -/
theorem ringMul_body_ops (σ : R) (z w : RingCell R) :
    ringMul σ z w = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- The multiplication's first slot is `z.1*w.1 + (σ*z.2)*w.2`, one addition over three
multiplications: the association the count reads. -/
theorem ringMul_body_first_slot (σ : R) (z w : RingCell R) :
    (ringMul σ z w).1 = z.1 * w.1 + (σ * z.2) * w.2 := rfl

/-- The multiplication's second slot is `z.1*w.2 + z.2*w.1`, one addition over two
multiplications. -/
theorem ringMul_body_second_slot (σ : R) (z w : RingCell R) :
    (ringMul σ z w).2 = z.1 * w.2 + z.2 * w.1 := rfl

/-- Unfolding the canonical norm form exhibits the three multiplications and one subtraction its
count names: `z.1^2` is the multiplication `z.1*z.1`, and `σ*z.2^2` is `σ*(z.2*z.2)`. -/
theorem ringNormForm_body_ops (σ : R) (z : RingCell R) :
    ringNormForm σ z = z.1 * z.1 - σ * (z.2 * z.2) := by
  simp only [ringNormForm, pow_two]

/-- Unfolding the channel read exhibits the one multiplication and one addition its count names. -/
theorem ringUpper_body_ops (r : R) (z : RingCell R) : ringUpper r z = z.1 + r * z.2 := rfl

/-! ### The extracted bodies and the one count they do not share

`cell.Cell.mul` and `cell.Cell.norm_form` are the generated bodies of `src/cell.rs:127-133` and
`:160-163`; `word_mul` and `word_norm` (`ApeironWord.lean:24082,24091`) are monomorphic wrappers
around them, and `word_mul_denote`/`word_norm_denote` (`WordRefinement.lean:45,56`) prove they agree
with the canonical definitions in value. On the arithmetic counts the two pairs agree exactly: both
multiplications perform five multiplications and two additions, both norm forms three multiplications
and one subtraction. The one field that differs is `cast`: the generated body reads its dial out of
the `Sigma` enum and converts it into the scalar ring (`ApeironWord.lean:133` and `:118`, one
`from_i64` each), whereas the canonical definitions receive `σ` already in `R` and convert nothing.

The counts of the two generated bodies are read off generated text. That reading is grade `model`,
exactly as `CostBridge`'s per-visit constants are (`CostBridge.lean:28-31`) and as
`RowDotAdmission`'s `srcOps`/`srcReads` are (`RowDotAdmission.lean:43-45`); the two canonical counts
of §2 are read off `Lean` definition bodies in this module. The comparisons below are `proved`,
because both records are closed terms. -/

/-- The generated `Cell<i64>` multiplication's own count: five `scalarRingInst.mul` applications
(`ApeironWord.lean:134,135,136,138,139`), two `add` (`:137,140`) and one dial conversion
(`from_i64`, `:133`). -/
def wordMulCost (σ : R) : RingCost σ := ⟨5, 2, 0, 0, 1, 0⟩

/-- The generated `Cell<i64>` norm form's own count: three `mul` (`ApeironWord.lean:119,120,121`),
one `sub` (`:122`) and one dial conversion (`from_i64`, `:118`). -/
def wordNormCost (σ : R) : RingCost σ := ⟨3, 0, 1, 0, 1, 0⟩

/-- **The generated multiplication performs the same arithmetic as the canonical definition, plus
exactly one dial conversion.** This is the count side of the correspondence `word_mul_denote`
proves on the value side: the same five multiplications, the same two additions, and the one
operation the generated body pays because its dial is an enum variant. -/
theorem wordMulCost_eq (σ : R) : wordMulCost σ = ringMulCost σ + (⟨0, 0, 0, 0, 1, 0⟩ : RingCost σ) := by
  simp [wordMulCost, ringMulCost, RingCost.composed]

/-- **The generated norm form performs the same arithmetic as the canonical definition, plus
exactly one dial conversion.** The same three multiplications, the same one subtraction. -/
theorem wordNormCost_eq (σ : R) : wordNormCost σ = ringNormFormCost σ + (⟨0, 0, 0, 0, 1, 0⟩ : RingCost σ) := by
  simp [wordNormCost, ringNormFormCost, RingCost.composed]

/-- The multiplication correspondence is exact on all six counts: five fields equal, and the dial
conversion differs by exactly one. -/
theorem wordMulCost_diff (σ : R) :
    (wordMulCost σ).mul - (ringMulCost σ).mul = 0 ∧ (wordMulCost σ).add - (ringMulCost σ).add = 0 ∧
      (wordMulCost σ).sub - (ringMulCost σ).sub = 0 ∧ (wordMulCost σ).neg - (ringMulCost σ).neg = 0 ∧
      (wordMulCost σ).channel - (ringMulCost σ).channel = 0 ∧
      (wordMulCost σ).cast - (ringMulCost σ).cast = 1 := by
  simp [wordMulCost, ringMulCost, RingCost.composed]

/-- The norm-form correspondence is exact on all six counts, on the same five fields and the same
dial conversion. -/
theorem wordNormCost_diff (σ : R) :
    (wordNormCost σ).mul - (ringNormFormCost σ).mul = 0 ∧
      (wordNormCost σ).add - (ringNormFormCost σ).add = 0 ∧
      (wordNormCost σ).sub - (ringNormFormCost σ).sub = 0 ∧
      (wordNormCost σ).neg - (ringNormFormCost σ).neg = 0 ∧
      (wordNormCost σ).channel - (ringNormFormCost σ).channel = 0 ∧
      (wordNormCost σ).cast - (ringNormFormCost σ).cast = 1 := by
  simp [wordNormCost, ringNormFormCost, RingCost.composed]

/-! ## 3. Composition over an iterated construction

The iterated construction is the corpus's own: `ringFold σ zs = zs.foldr (ringMul σ) ringOne`
(`ChannelsAnyRing.lean:498`), the product of a list of cells of the algebra. The corpus describes it
as `Gates.fold` over the reals and states three facts about it that this section turns into costs:

- `ringUpper_fold` (`ChannelsAnyRing.lean:509`) — the first-coordinate read of the fold is the
  product of the first-coordinate reads of the elements. A fold that reads the channel therefore
  pays exactly **one `ringUpper` per element**, which is the `channel` field of the link cost.
- `ringLower_fold` (`:524`) — the same for the second coordinate.
- `ringNormForm_fold` (`:547`) — the size of the fold is the product of the two channel products.
  So the two reads are per-element operations, not one read at the end.

`ringMul` is associative and commutative (`AnyRing.lean:191,180`), so the fold's value does not
depend on its bracketing; and `RingCost.composed_assoc`/`composed_comm` (§1) say the same of the
measure. The recurrence below is therefore stated left-associated and applies to the corpus's
right-associated fold alike.

### The iteration operator, the recurrence, and the closure

**Measure.** `RingCost σ`, six fields, one counted quantity each (§1).

**Counted quantity.** Scalar ring operations of the algebra: multiplications, additions,
subtractions, negations, dial conversions, channel reads. No memory, no dispatch, no timing.

**`RingCost.rep` is the iteration operator.** `rep σ n a` is the cost of `n` dependent copies of an
operation of cost `a`. It is a closed term — six `Nat` multiplications, no recursion — so its own
temporal and spatial cost are O(1): **the growth is in the counted quantity, not in the work the
function does.** That distinction is the reason §8's "not immediate from its type" clause matters
here, and it is why `rep` is a cost and not an algorithm.

**Recurrence.** `rep σ 0 a = zero σ` and `rep σ (n + 1) a = a + rep σ n a` (`RingCost.rep_zero`,
`RingCost.rep_succ` below): one copy per step, plus whatever the predecessor has spent.

**Why it closes.** The recursion is on `Nat`, so the induction is well founded with a closed base
case; `RingCost.rep_add` proves the recurrence telescopes, which is the closure statement: at `n`
copies every field is finite and equals `n` times the corresponding field of one copy, and the
grouping of the `n`-fold sum is irrelevant. -/

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.**
Temporal O(1) and spatial O(1): six `Nat` multiplications on a six-field record, no recursion, no
allocation. Counted in: nothing — this is the iteration operator of the measure, not an operation
of the algebra; the `n` it is given is the length of a repeated *operation*, so the value grows
linearly while the function does not recur. Derivation: fieldwise `n * a.field`, which is the
componentwise image of the `n`-fold dependent sum proved additive in `RingCost.rep_add`. -/
def RingCost.rep (σ : R) (n : Nat) (a : RingCost σ) : RingCost σ :=
  ⟨n * a.mul, n * a.add, n * a.sub, n * a.neg, n * a.cast, n * a.channel⟩

/-- **Base case: zero copies cost nothing.** -/
@[simp] theorem RingCost.rep_zero (σ : R) (a : RingCost σ) :
    RingCost.rep σ 0 a = RingCost.zero σ := by
  simp [RingCost.rep, RingCost.zero]

/-- **Recurrence: one more copy is one copy plus the measure of the copies already made.** -/
theorem RingCost.rep_succ (σ : R) (n : Nat) (a : RingCost σ) :
    RingCost.rep σ (n + 1) a = a + RingCost.rep σ n a := by
  cases a
  simp [RingCost.rep, RingCost.composed, Nat.succ_mul, Nat.add_comm]

/-- **THE COMPOSITION LAW over an iterated construction.** `n + m` copies cost the sum of `n` copies
and `m` copies, field by field. Together with `rep_succ` this is additivity of the measure over an
iterated construction stated once, for every operation of the algebra: it is the ring-operation
counterpart of `CostBridge.toCost_add` (`CostBridge.lean:170`), which states the same law for the
registered schedule record. -/
theorem RingCost.rep_add (σ : R) (n m : Nat) (a : RingCost σ) :
    RingCost.rep σ (n + m) a = RingCost.rep σ n a + RingCost.rep σ m a := by
  cases a
  simp [RingCost.rep, RingCost.composed, Nat.add_mul]

/-- Iteration distributes over dependent composition: `n` copies of two composed operations cost
`n` copies of each. This is the law that lets a link cost be stated once and iterated. -/
theorem RingCost.rep_composed (σ : R) (n : Nat) (a b : RingCost σ) :
    RingCost.rep σ n (a + b) = RingCost.rep σ n a + RingCost.rep σ n b := by
  cases a
  cases b
  simp [RingCost.rep, RingCost.composed, Nat.mul_add]

/-! ### The corpus fold and its measure -/

/-- **The measure of the canonical iterated construction** `ringFold σ zs`
(`ChannelsAnyRing.lean:498`). Temporal O(|zs|) — one record composition per element, by structural
recursion on the list; spatial O(1) words beyond the input and the result record (an accumulator
and one frame per element, and `List.foldr`'s own frame is `|zs|` deep, so the honest spatial
statement is O(|zs|) stack frames and O(1) *result* words). Counted in: the six `RingCost` fields,
one per operation kind (§1). Derivation: `foldr f init [] = init` performs no `f`, and
`foldr f init (z :: zs) = f z (foldr f init zs)` performs exactly one `f`, so one `ringMul` — that
is one `ringMulCost σ` — per element. Closed by `ringFoldCost_eq` below, which is `|zs|` copies of
the link. -/
def ringFoldCost (σ : R) : List (RingCell R) → RingCost σ
  | [] => RingCost.zero σ
  | _ :: zs => ringMulCost σ + ringFoldCost σ zs

/-- **Closure of the fold's recurrence: the measure of a fold is one multiplication per element.**
By induction on the list; `[]` is the base case and a cons is one `ringMul` on top of the fold of
the tail. -/
theorem ringFoldCost_eq (σ : R) (zs : List (RingCell R)) :
    ringFoldCost σ zs = RingCost.rep σ zs.length (ringMulCost σ) := by
  induction zs with
  | nil => simp [ringFoldCost, RingCost.rep, ringMulCost, RingCost.zero]
  | cons z zs ih => rw [ringFoldCost, ih, List.length_cons, RingCost.rep_succ,
      RingCost.composed_comm]

/-- The empty fold is free, matching `foldr`'s base case `ringOne`, whose cost is not an operation. -/
theorem ringFoldCost_nil (σ : R) : ringFoldCost σ [] = RingCost.zero σ := rfl

/-- One element of a fold is one multiplication, stated on the record directly. -/
theorem ringFoldCost_cons (σ : R) (z : RingCell R) (zs : List (RingCell R)) :
    ringFoldCost σ (z :: zs) = ringMulCost σ + ringFoldCost σ zs := rfl

/-- **The per-element channel read of a fold is the corpus's own statement, used rather than
assumed.** `ringUpper_fold` (`ChannelsAnyRing.lean:509`) says the channel read of the fold equals
the product of the per-element channel reads, so counting one `ringUpperCost` per element counts
the operations the corpus says a channel fold performs. -/
theorem ringFold_channel_is_elementwise (σ r : R) (hr : r ^ 2 = σ) (zs : List (RingCell R)) :
    ringUpper r (ringFold σ zs) = (zs.map (ringUpper r)).prod :=
  ringUpper_fold hr zs

/-- The same for the second coordinate (`ringLower_fold`, `ChannelsAnyRing.lean:524`). -/
theorem ringFold_lower_is_elementwise (σ r : R) (hr : r ^ 2 = σ) (zs : List (RingCell R)) :
    ringLower r (ringFold σ zs) = (zs.map (ringLower r)).prod :=
  ringLower_fold hr zs

/-- **The measure of a two-channel fold: the fold's own multiplications plus one read of each
channel per element.** `|zs|` multiplications' worth of work, then `|zs|` first-coordinate reads
and `|zs|` second-coordinate reads. This is the cost of what `ringNormForm_fold` (`:547`) states:
two running products and one final multiplication per channel, over `|zs|` elements. -/
theorem twoChannelFoldCost_eq (σ : R) (zs : List (RingCell R)) :
    ringFoldCost σ zs + RingCost.rep σ zs.length (ringUpperCost σ) +
        RingCost.rep σ zs.length (ringLowerCost σ)
      = RingCost.rep σ zs.length
          (ringMulCost σ + ringUpperCost σ + ringLowerCost σ) := by
  calc
    ringFoldCost σ zs + RingCost.rep σ zs.length (ringUpperCost σ) +
          RingCost.rep σ zs.length (ringLowerCost σ)
        = RingCost.rep σ zs.length (ringMulCost σ)
            + (RingCost.rep σ zs.length (ringUpperCost σ)
              + RingCost.rep σ zs.length (ringLowerCost σ)) := by
            rw [ringFoldCost_eq, RingCost.composed_assoc]
    _ = RingCost.rep σ zs.length (ringMulCost σ + ringUpperCost σ)
          + RingCost.rep σ zs.length (ringLowerCost σ) := by
          rw [RingCost.rep_composed]
          exact (RingCost.composed_assoc σ _ _ _).symm
    _ = RingCost.rep σ zs.length (ringMulCost σ + ringUpperCost σ + ringLowerCost σ) :=
        (RingCost.rep_composed σ _ _ _).symm

/-! ### The length-indexed link chain -/

/-- **One link of a chain:** a multiplication followed by a first-coordinate read, that is
`⟨6, 3, 0, 0, 0, 1⟩`. The channel read is what the corpus's `ringUpper_fold` attributes to each
element, so it belongs in the link rather than at the end of the chain. -/
def linkCost (σ : R) : RingCost σ := ringMulCost σ + ringUpperCost σ

/-- **The cost of a chain of `n` links.** Temporal `n` recursive steps, each one record addition;
spatial one accumulator and one stack frame, O(1) words beyond the result record. Counted in: the
six `RingCost` fields. Derivation: the recurrence `c 0 = zero σ`, `c (n + 1) = c n + linkCost σ`,
one link per step; it closes because `RingCost.rep_add` telescopes it to `n` copies of the link
(`chainMulCost_rep` below) and `Nat` induction has no open base case. -/
def chainMulCost (σ : R) : Nat → RingCost σ
  | 0 => RingCost.zero σ
  | n + 1 => chainMulCost σ n + linkCost σ

/-- **Additivity over the iterated construction, in the measure's own iteration operator.** The
chain of `n` links costs `n` copies of the link's cost. -/
theorem chainMulCost_rep (σ : R) (n : Nat) : chainMulCost σ n = RingCost.rep σ n (linkCost σ) := by
  induction n with
  | zero => simp [chainMulCost, RingCost.rep_zero, RingCost.zero]
  | succ n ih => rw [chainMulCost, ih, RingCost.rep_succ, RingCost.composed_comm]

/-- **The same statement with the numbers substituted:** `n` links are `n * 6` multiplications,
`n * 3` additions and `n` channel reads, because one link is `ringMulCost + ringUpperCost =
⟨6, 3, 0, 0, 0, 1⟩`. This is the kernel-checked composition statement the chain needs: the measure
of a construction of length `n` is the sum of the measures of its `n` links, and the grouping of
that sum is irrelevant (`RingCost.composed_assoc` and `RingCost.composed_comm` are the two grouping
laws). -/
theorem chainMulCost_eq (σ : R) (n : Nat) :
    chainMulCost σ n = (⟨n * 6, n * 3, 0, 0, 0, n⟩ : RingCost σ) := by
  have h : RingCost.rep σ n (linkCost σ) = (⟨n * 6, n * 3, 0, 0, 0, n⟩ : RingCost σ) := by
    cases n <;>
      simp [RingCost.rep, linkCost, ringMulCost, ringUpperCost, RingCost.composed, RingCost.zero,
        Nat.succ_mul, Nat.mul_one, Nat.one_mul]
  rw [chainMulCost_rep, h]

/-- **Closure of the recurrence in the fieldwise form:** every field of the chain's cost is the
number of links times the field of one link, i.e. the measure of the fold plus the measure of the
per-element channel reads. This is the form `CostAlgebra.planCost_shared_add` (`CostAlgebra.lean:379`)
takes for a plan's shared effect, read over ring operations: no field of the chain escapes the
record. -/
theorem chainMulCost_fields (σ : R) (n : Nat) :
    chainMulCost σ n = RingCost.rep σ n (ringMulCost σ) + RingCost.rep σ n (ringUpperCost σ) := by
  rw [chainMulCost_rep, linkCost, RingCost.rep_composed]

/-- **The chain of links and the canonical fold differ in exactly the channel reads the chain
adds.** Both perform `n` multiplications' worth of ring multiplication; the chain pays `n` extra
multiplications, `n` extra additions and `n` channel reads because each link also reads the
channel, and the fold pays none of those. This is the honest relation between the ad-hoc chain and
the corpus's `ringFold`: they are two different constructions, and the record separates them. -/
theorem chainMulCost_vs_fold (σ : R) (n : Nat) :
    (chainMulCost σ n).mul = (RingCost.rep σ n (ringMulCost σ)).mul + n ∧
      (chainMulCost σ n).add = (RingCost.rep σ n (ringMulCost σ)).add + n ∧
      (chainMulCost σ n).channel = n ∧ (RingCost.rep σ n (ringMulCost σ)).channel = 0 ∧
      (chainMulCost σ n).sub = (RingCost.rep σ n (ringMulCost σ)).sub := by
  rw [chainMulCost_eq]
  simp [RingCost.rep, ringMulCost] <;> omega

/-- The chain's cost grows strictly with its length in the multiplication field, so a longer chain
is never free: the measure is not degenerate on the iterated construction. -/
theorem chainMulCost_strict (σ : R) {m n : Nat} (h : m < n) :
    (chainMulCost σ m).mul < (chainMulCost σ n).mul := by
  simp only [chainMulCost_eq]
  omega

/-- **Composition of chains is addition of their lengths.** Two chains of lengths `m` and `n`
composed one after the other cost the sum, so the measure of the concatenation is the sum of the
measures: `RingCost.rep_add` instantiated at the link. -/
theorem chainMulCost_add (σ : R) (m n : Nat) :
    chainMulCost σ (m + n) = chainMulCost σ m + chainMulCost σ n := by
  simp only [chainMulCost_eq]
  apply RingCost.eq_of_fields <;> simp [RingCost.composed, Nat.add_mul, Nat.mul_add]

/-- **A chain of length one is one multiplication and one channel read.** The base case and the
first step, so the induction of `chainMulCost_rep` has a checked instance at each end. -/
theorem chainMulCost_one (σ : R) : chainMulCost σ 1 = linkCost σ := by
  simp [chainMulCost, RingCost.zero, RingCost.composed]

/-- **A chain of length zero is free.** -/
theorem chainMulCost_zero (σ : R) : chainMulCost σ 0 = RingCost.zero σ := by
  rfl

/-! ## 4. The crypto-relevant operations

Two operations are counted, one from each end of the evidence. `wrapping_add` is the operation the
extracted word kernel performs and whose denotation is already proved (`Denotation.lean:35`). The
rotation is the operation the two-slot reading of a word rotation is made of
(`work/algebra-rooting/ops-map.md:94` row S7, `:111` row K2, `:113` row K4); its *denotation* is the
candidate `C5` at grade `model` there and is not proved, so it is carried below as the named premise
`RotRead` and every cost statement stays unconditional. -/

/-- **The denotation of one extracted wrapping addition.** The single operation
`core.num.I64.wrapping_add` denotes the ring addition of the two words at width 64
(`Denotation.lean:35-38`), and the measure of that operation is `wrappingAddCost`, one ring
addition. This is the crypto row whose denotation link is `proved` rather than `model`. -/
theorem wrappingAdd_cost_denote (x y : I64) :
    denote (core.num.I64.wrapping_add x y) = denote x + denote y ∧
      (wrappingAddCost (denoteSigma cell.Sigma.Hyperbolic)).add = 1 ∧
        (wrappingAddCost (denoteSigma cell.Sigma.Hyperbolic)).mul = 0 :=
  ⟨denote_wrapping_add x y, rfl, rfl⟩

/-- **The generated multiplication's cost, anchored on its proved denotation.** Two facts in one
statement: `word_mul_denote` (`WordRefinement.lean:45`) says the extracted body computes the
canonical `ringMul`, and the count says it does so with five multiplications, two additions and one
dial conversion — the five and two of `ringMulCost` plus exactly one `cast`, which is the
`from_i64` application at `ApeironWord.lean:133`. The cast is the only counted operation whose
*value* is free: `denote_from_i64` (`Denotation.lean:29`) proves the extraction of an `i64` into
the scalar ring denotes the identity. So the measure counts an operation, not a cost. -/
theorem wordMulCost_denote (a : cell.Cell I64) (s : cell.Sigma) (b : cell.Cell I64) :
    (do let c ← word_mul a s b; ok (denoteCell c)) =
        ok (ringMul (denoteSigma s) (denoteCell a) (denoteCell b)) ∧
      (wordMulCost (denoteSigma s)).cast = 1 ∧
        (wordMulCost (denoteSigma s)).mul = (ringMulCost (denoteSigma s)).mul := by
  refine ⟨word_mul_denote a s b, ?_, ?_⟩
  · rfl
  · simp [wordMulCost, ringMulCost, RingCost.composed]

/-- **The extracted integer cast is free in value and is counted as one operation.** The one
`cast` in `wordMulCost` and in `wordNormCost` is this application, and its denotation is the
identity, so the count's extra unit is a real operation of the body and not a value-changing step. -/
theorem cast_denote_identity (v : I64) :
    (do let r ← I64.Insts.Apeiron_wordScalarRing.from_i64 v; ok (denote r)) = ok (denote v) ∧
      (wordMulCost (denoteSigma cell.Sigma.Hyperbolic)).cast = 1 := by
  exact ⟨denote_from_i64 v, rfl⟩

/-- **The generated norm form's cost, anchored on its proved denotation.** `word_norm_denote`
(`WordRefinement.lean:56`) says the extracted body computes the canonical `ringNormForm`; the count
is three multiplications, one subtraction and one dial conversion, the arithmetic of
`ringNormFormCost` plus exactly one `cast`. -/
theorem wordNormCost_denote (a : cell.Cell I64) (s : cell.Sigma) :
    (do let r ← word_norm a s; ok (denote r)) =
        ok (ringNormForm (denoteSigma s) (denoteCell a)) ∧
      (wordNormCost (denoteSigma s)).cast = 1 ∧
        (wordNormCost (denoteSigma s)).sub = (ringNormFormCost (denoteSigma s)).sub := by
  refine ⟨word_norm_denote a s, ?_, ?_⟩
  · rfl
  · simp [wordNormCost, ringNormFormCost, RingCost.composed]

/-- **The rotation's reading, as a named premise.** `RotRead` is the candidate statement
`work/algebra-rooting/ops-map.md:167-180` (`rotl64_upper_mul`): the channel read of the product
`ringMul 1 (x,0) (2^n, 2^(w-n))` is the left rotation of `x` by `n` bits. It is stated here as the
premise of `rotation_cost_of_reading` and is not proved by this module. -/
def RotRead (w n : Nat) (x : ZMod (2 ^ w)) : Prop :=
  ringUpper 1 (ringMul 1 (x, 0) ((2 ^ n : Nat), (2 ^ (w - n) : Nat))) =
    (((x.val * (2 ^ n)) % (2 ^ w) + x.val / (2 ^ (w - n)) : Nat) : ZMod (2 ^ w))

/-- **The cost of one word rotation under the two-slot reading**, with the reading carried as the
named premise `RotRead`. The operation is one canonical multiplication followed by one channel read,
so its cost is `ringMulCost + ringUpperCost` and nothing else; `_h` is unused, which is the point:
the cost does not wait on the denotation. The *identification* of this operation with a word
rotation is `RotRead`, grade `model` at `work/algebra-rooting/ops-map.md:167-180`, and is not
discharged here. -/
theorem rotation_cost_of_reading (σ : R) (w n : Nat) (x : ZMod (2 ^ w)) (_h : RotRead w n x) :
    (rotationCost σ).mul = 6 ∧ (rotationCost σ).add = 3 ∧
      (rotationCost σ).channel = 1 ∧ ringMulCost σ + ringUpperCost σ = rotationCost σ := by
  exact ⟨rotationCost_fields σ ▸ rfl, rotationCost_fields σ ▸ rfl,
    rotationCost_fields σ ▸ rfl, rfl⟩

/-- **The per-face rotation cost at a crypto width.** At the hyperbolic face `σ = 1` and at the
parabolic face `σ = 0` the same expression has the same count: six multiplications, three additions,
one channel read, no dial conversion. The cost is the count of the *bodies*, and no counted body
branches on the dial, so the per-face costs of this rotation agree at these two faces; the dial
enters the `ringMul` result, not the count. This is the derived per-face cost the rooting wave
carries, and it is the honest form of a per-face claim: equal counts, not a missing case. -/
theorem rotation_cost_per_face (w : Nat) :
    (rotationCost (1 : ZMod (2 ^ w))).mul = (rotationCost (0 : ZMod (2 ^ w))).mul ∧
      (rotationCost (1 : ZMod (2 ^ w))).add = (rotationCost (0 : ZMod (2 ^ w))).add ∧
      (rotationCost (1 : ZMod (2 ^ w))).sub = (rotationCost (0 : ZMod (2 ^ w))).sub ∧
      (rotationCost (1 : ZMod (2 ^ w))).neg = (rotationCost (0 : ZMod (2 ^ w))).neg ∧
      (rotationCost (1 : ZMod (2 ^ w))).cast = (rotationCost (0 : ZMod (2 ^ w))).cast ∧
      (rotationCost (1 : ZMod (2 ^ w))).channel = (rotationCost (0 : ZMod (2 ^ w))).channel ∧
      (rotationCost (1 : ZMod (2 ^ w))).mul = 6 := by
  simp [rotationCost, ringMulCost, ringUpperCost, RingCost.composed]

/-! ## 5. Worked numeric examples

A chain of three multiplications with a channel read at each link. The counts are `Nat` values in
the record; each is decided, so the numbers are the measure's and not an estimate. -/

/-- **Worked example, length three.** Three links, each one multiplication and one channel read:
one link is `⟨5, 2, 0, 0, 0, 0⟩ + ⟨1, 1, 0, 0, 0, 1⟩ = ⟨6, 3, 0, 0, 0, 1⟩`, so three links are
`3 * 6 = 18` multiplications, `3 * 3 = 9` additions, three channel reads, and no dial conversion,
because the canonical definitions receive `σ` in the ring. -/
theorem chain3_worked (σ : R) : chainMulCost σ 3 = (⟨18, 9, 0, 0, 0, 3⟩ : RingCost σ) :=
  chainMulCost_eq σ 3

/-- The same example at length one: six multiplications, three additions, one channel read. -/
theorem chain1_worked (σ : R) : chainMulCost σ 1 = (⟨6, 3, 0, 0, 0, 1⟩ : RingCost σ) :=
  chainMulCost_eq σ 1

/-- The same example at length zero: nothing. -/
theorem chain0_worked (σ : R) : chainMulCost σ 0 = (⟨0, 0, 0, 0, 0, 0⟩ : RingCost σ) :=
  chainMulCost_zero σ

/-- **A second worked example at a concrete small width, on the corpus's own iterated
construction.** At `w = 4`, that is `ZMod 16`, the fold of the three cells `(2,0)`, `(0,3)`, `(1,1)`
has the same count as at width 64, because no counted body depends on `w`. The count and the value
are separate facts and are stated separately: the count is a count of bodies, the value is an
element of the ring. -/
def smallList : List (RingCell (ZMod 16)) := [(2, 0), (0, 3), (1, 1)]

/-- The worked fold: `ringFold` of `smallList` at the hyperbolic face of `ZMod 16`. -/
def smallFold : RingCell (ZMod 16) := ringFold (1 : ZMod 16) smallList

/-- **The fold's count, three elements.** Three links' worth: eighteen multiplications, nine
additions, three channel reads, no dial conversion, because the canonical definitions receive `σ`
in the ring. -/
theorem smallFold_cost :
    chainMulCost (1 : ZMod 16) 3 = (⟨18, 9, 0, 0, 0, 3⟩ : RingCost (1 : ZMod 16)) :=
  chainMulCost_eq _ 3

/-- **The same fold, costed as the corpus fold itself rather than as a chain of links:** three
multiplications and no channel read, because `ringFold` performs no `ringUpper`. -/
theorem smallFold_fold_cost :
    ringFoldCost (1 : ZMod 16) smallList = (⟨15, 6, 0, 0, 0, 0⟩ : RingCost (1 : ZMod 16)) := by
  rw [ringFoldCost_eq]
  simp [RingCost.rep, ringMulCost, smallList]

/-- **The fold's value at `ZMod 16`, decided.** `ringFold` is right-associated from the unit:
`(2,0) * ((0,3) * ((1,1) * (1,0)))`, which is `(6,6)` — computed over the sixteen elements of the
ring, so this is the measure's ring and the corpus's fold together, not a transcription. -/
theorem smallFold_value : smallFold = (6, 6) := by
  decide

/-- **The fold's two channel reads and its size at `ZMod 16`, decided.** `ringUpper 1 (6,6) = 12`
(the first coordinate along `(1,1)`), `ringLower 1 (6,6) = 0` (the second), and the size
`6^2 - 1 * 6^2 = 0`, which agrees with `ringNormForm_fold` (`ChannelsAnyRing.lean:547`): the
channel products are `2 * 3 * 2 = 12` and `2 * 3 * 0 = 0`. The zero second coordinate is a fact
about this particular list, not an omission: `(1,1)` has lower read `1 - 1 = 0`, which is exactly
the case `ringUpper_ringIdemMinus` (`ChannelsAnyRing.lean:404`) names. -/
theorem smallFold_channels :
    ringUpper (1 : ZMod 16) smallFold = 12 ∧ ringLower (1 : ZMod 16) smallFold = 0 ∧
      ringNormForm (1 : ZMod 16) smallFold = 0 := by
  decide

/-! ## 6. The bounds this measure does and does not give

`CostAlgebra.registeredCost` (`:255`) collapses its record to one scalar for an admission
certificate. This module keeps the record and states the correspondence instead, because a scalar
hides a regression in one count (`CostAlgebra.scalar_not_sufficient`, `:271`) — the same blindness
`CostBridge.loads_invisible_to_order` (`:161`) exhibits for element traffic. What is **not** given:
any bound in time, latency, bytes, energy or dispatches, and any statement about hardware. -/

/-- **No total order by a single number is asserted.** Two records tie on four counts and differ on
two, so any scalar reading all six fields strictly would be wrong, and the record is the measure. -/
theorem RingCost.not_linear (σ : R) :
    ∃ a b : RingCost σ, a.mul = b.mul ∧ a.add < b.add ∧ a.channel < b.channel := by
  exact ⟨(⟨0, 0, 0, 0, 0, 0⟩ : RingCost σ), (⟨0, 1, 0, 0, 0, 1⟩ : RingCost σ), rfl, by simp,
    by simp⟩

/-- The measure is monotone in the chain length: replacing a chain by a no-longer one never raises
the cost, in the multiplication field. -/
theorem chainMulCost_mono (σ : R) {m n : Nat} (h : m ≤ n) :
    (chainMulCost σ m).mul ≤ (chainMulCost σ n).mul := by
  simp only [chainMulCost_eq]
  omega

/-! ## 7. Audit rows

Every headline above, printed with its axiom set. The permitted cone is
`{propext, Classical.choice, Quot.sound}` and no row may contain `sorryAx`. -/

#print axioms RingCost.eq_of_fields
#print axioms RingCost.composed_comm
#print axioms RingCost.composed_assoc
#print axioms RingCost.composed_zero
#print axioms RingCost.le_of_fields
#print axioms RingCost.le_refl'
#print axioms RingCost.le_trans'
#print axioms RingCost.le_antisymm'
#print axioms RingCost.rep_zero
#print axioms RingCost.rep_succ
#print axioms RingCost.rep_add
#print axioms RingCost.rep_composed
#print axioms rotationCost_fields
#print axioms ringMul_body_ops
#print axioms ringMul_body_first_slot
#print axioms ringMul_body_second_slot
#print axioms ringNormForm_body_ops
#print axioms ringUpper_body_ops
#print axioms ringLower_body_ops
#print axioms wordMulCost_eq
#print axioms wordNormCost_eq
#print axioms wordMulCost_diff
#print axioms wordNormCost_diff
#print axioms wordMulCost_denote
#print axioms wordNormCost_denote
#print axioms cast_denote_identity
#print axioms ringFoldCost_eq
#print axioms ringFoldCost_nil
#print axioms ringFoldCost_cons
#print axioms ringFold_channel_is_elementwise
#print axioms ringFold_lower_is_elementwise
#print axioms twoChannelFoldCost_eq
#print axioms chainMulCost_rep
#print axioms chainMulCost_eq
#print axioms chainMulCost_fields
#print axioms chainMulCost_vs_fold
#print axioms chainMulCost_strict
#print axioms chainMulCost_add
#print axioms chainMulCost_one
#print axioms chainMulCost_zero
#print axioms wrappingAdd_cost_denote
#print axioms rotation_cost_of_reading
#print axioms rotation_cost_per_face
#print axioms chain3_worked
#print axioms chain1_worked
#print axioms chain0_worked
#print axioms smallFold_cost
#print axioms smallFold_fold_cost
#print axioms smallFold_value
#print axioms smallFold_channels
#print axioms RingCost.not_linear
#print axioms chainMulCost_mono

end algebra_cost
