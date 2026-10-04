/-
SCOPE: work/proofs/AlgebraCostKeccak.lean, new, and its receipt work/proofs/AlgebraCostKeccak.md. The
cost of the Keccak-f[1600] round and permutation and of one SHA3-256 absorb block in AR-15/AR-20's
eleven-field `RoundCost` measure, with a second two-field record for the lane-layer quantities the
eleven fields cannot express (the pi lane permutation and the byte-to-lane widening of the sponge
injection); the per-body counts read off `Crypto/Keccak/Round.lean` and `Crypto/Sponge.lean`; the
24-round closed form with additivity; the absorb-step figure at the SHA3-256 rate 136; and the
SHA-256-attempt / Keccak-permutation / SHA3-absorb comparison with the face verdict and the
generality statement. Complete when the module elaborates with every headline audited inside
{propext, Classical.choice, Quot.sound} and zero sorryAx, seeded count defects each reject at
attributable theorems, and the receipt records the counts, their grades, the named premises and the
residual list.
REGISTER: algebra/resource model. No production, pin, API, extraction-root, corpus or gate-driver
edit; no Rust source; no cargo; no regen.sh; no measurement. Excluded by construction: latency,
wall-clock time, throughput, hash rate, bytes per second, energy, joules, watts, hardware, compiler
or codegen cost, the kernel ABI, the `unsafe` container, device dispatch cost, constant-time and
security properties of any hash, and every asymptotic statement. AGENTS.md:429 - a cost-model proof
is not a hardware speed measurement; this module states counts and nothing else.
CORPUS: `Crypto.Keccak.Round.{lane:23,column:27,theta:32,rho:38,pi:42,chi:49,iota:57,round:61,
permute:65}` and `Crypto.Keccak.Constants.{RC:8,ROT:17}` at repos/apeiron/proof/corpus/Crypto/Keccak/,
sha256 Round.lean c0a4d608707dbe109f97b50daa4e28a80e274aaf5b716dba71f4fb137a737e18, Constants.lean
44483feb6607e8fdf50be79a275d523de472730be3e0911fd4d5d7db2ed8f016;
`Crypto.Sponge.{rate_sha3_256:29,xorBlock:55,absorb:85,sha3_256:104}` at
repos/apeiron/proof/corpus/Crypto/Sponge.lean, sha256
9a4eabcca891114609314ae893a933fb071da4dbf3052020045a06a170ec5973; both are roots of the `Crypto`
corpus `lean_lib` (proof/corpus/lakefile.lean:137-138). The ring bodies are
`Apeiron.Trichotomy.AnyRing.{RingCell:98,ringMul:109,ringUpper:139}` at the pinned revision
880aa57e47e782ed7a9d9ee1974cfc87360b446b through
repos/apeiron/proof/corpus/.lake/packages/apeiron/proof/, grade `proved` there.
REFERENCE: repos/apeiron/AGENTS.md:336-341 (section 8 per-function complexity: the measure, the
recurrence, why it closes), :156-167 and :226-240 (section 4 obligations and section 5 grades), :429
(a cost-model proof is not a hardware speed measurement). work/proofs/AlgebraCostRound.lean/.md
(AR-15), sha256 908c7aa729a14164a050490a67b78de83e92398fc72e2e4fa894c8077bdc6c8b: `RoundCost`
(eleven fields), `roundCost`, `blockCost`, `RoundCost.rep_add`. work/proofs/AlgebraCostPow.lean/.md
(AR-20), sha256 a3d62397ffd67d6902c8b28629af42d394df3c55bb176558d8f353b1694207e0: `attemptCost`,
`attemptCost_eq`, `blockCost_eq`, `RoundCost.rep`, `RingCost.rep`. work/algebra-rooting/ops-map.md:
:100-122 (Keccak rows K1-K13), :136-146 (obstructions O1-O4), :167-180 (candidate C5, the rotation
reading).
GATE: lane lake env lean
~/icarus/work/proofs/AlgebraCostKeccak.lean with cwd
proof/corpus. Wall-clock budget
590 s for the invocation and its controls, including queue time, under
flock /tmp/apeiron/apeiron-lane.lock, MemoryMax=3800M, MemoryHigh=3400M, MemorySwapMax=0,
LEAN_NUM_THREADS=1. No cargo, no lake build of the package, no regen.sh, no nohup/setsid, no retry
loop, no raised maxRecDepth, no maxHeartbeats 0.
GRADE: `proved` for every operation count that is read off a Lean definition body by unfolding it in
this module - the ring bodies through `ringMul`/`ringUpper`, the Keccak bodies through
`Crypto.Keccak.Round` and `Crypto.Sponge` - and for the measure's laws, the permutation and absorb
closed forms, the additivity, the comparison arithmetic and the face equality; the cone is
{propext, Classical.choice, Quot.sound} with no sorryAx, checked by the `#print axioms` rows below.
`model` for the C5 identification of a rotation with `ringMul`-plus-`ringUpper`, carried as the named
premise `RotRead64` and never as an axiom. `tested`: none - no `cargo test` ran. Unmeasured and
unclaimed: everything on the REGISTER exclusion list.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sponge
import Mathlib.Tactic

set_option autoImplicit false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
set_option linter.unnecessarySeqFocus false

namespace algebra_cost_keccak

universe u

variable {R : Type u} [CommRing R]

open Apeiron.Crypto.Keccak
open Apeiron.Trichotomy.AnyRing

/-! ## 1. The measure

AR-15's `RoundCost σ` (eleven fields, one counted quantity each) and AR-10's `RingCost σ` (six ring
fields) are re-declared here exactly as in AR-20, because neither `work/proofs` module is a root of a
corpus `lean_lib` and neither can be imported. The counts of this module are stated in those records,
so a comparison against AR-20's attempt figure is field for field.

Keccak-f[1600] performs operations that fit no `RoundCost` field: the pi lane permutation relocates
lane values and the sponge injection widens bytes to lane words. Those are carried in a second,
two-field record `LaneLayer`, one counted quantity per field, rather than folded into an existing
field. §2 states the convention that separates a lane *move* from element traffic. -/

/-- `RoundCost σ`: AR-15/AR-20's measure, re-declared. Eleven fields, one counted quantity each. -/
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

/-- **Dependent composition.** Temporal and spatial O(1): a closed term of eleven `Nat` additions.
Counted in: nothing - this is the measure's own operator, not an operation of a hash. -/
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

/-- **Composition is commutative**: disjoint work sums the same in either order. -/
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

/-- The **cost vector**: the eleven counts in field order, so records at different dials compare. -/
def RoundCost.vector {σ : R} (c : RoundCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr)

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.** Temporal and
spatial O(1): eleven `Nat` multiplications, no recursion, no allocation. Counted in: nothing - this
is the iteration operator of the measure; the growth is in the counted quantity, not in the work the
definition does. Derivation: fieldwise `n * a.field`, proved additive in `RoundCost.rep_add`. -/
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

/-- **The composition law over an iterated construction, eleven fields.** `n + m` copies cost the sum
of `n` copies and `m` copies, field by field; every permutation or chain figure below inherits its
additivity from this theorem. -/
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

/-- `RingCost σ`: AR-10's six-field ring measure, re-declared. -/
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

/-- Dependent composition over the six ring fields. -/
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

@[simp] theorem RingCost.rep_zero (σ : R) (a : RingCost σ) :
    RingCost.rep σ 0 a = RingCost.zero σ := by
  simp [RingCost.rep, RingCost.zero]

/-- **The composition law over an iterated construction, six fields.** -/
theorem RingCost.rep_add (σ : R) (n m : Nat) (a : RingCost σ) :
    RingCost.rep σ (n + m) a = RingCost.rep σ n a + RingCost.rep σ m a := by
  cases a
  simp [RingCost.rep, RingCost.composed, Nat.add_mul]

theorem RingCost.rep_composed (σ : R) (n : Nat) (a b : RingCost σ) :
    RingCost.rep σ n (a + b) = RingCost.rep σ n a + RingCost.rep σ n b := by
  cases a
  cases b
  simp [RingCost.rep, RingCost.composed, Nat.mul_add]

/-- **The ring projection**: the first six fields of `RoundCost σ` read as AR-10's `RingCost σ`. -/
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

/-- **`LaneLayer`: the counted quantities of the Keccak bodies that no `RoundCost` field can
express.** `laneMove` counts an assignment of a lane value to a *different* lane index that performs
no value operation; `widen` counts an 8-bit byte extended to a 64-bit lane word. One counted quantity
per field; §2 states the derivation and the convention. -/
structure LaneLayer where
  /-- lane index moves: assignments of a lane value to a different lane index -/
  laneMove : Nat
  /-- byte widenings to a lane word -/
  widen : Nat
  deriving DecidableEq, Repr

/-- The record of a computation that performs neither lane-layer operation. -/
def LaneLayer.zero : LaneLayer := ⟨0, 0⟩

/-- Dependent composition over the two lane-layer fields. Temporal and spatial O(1). -/
def LaneLayer.composed (a b : LaneLayer) : LaneLayer :=
  ⟨a.laneMove + b.laneMove, a.widen + b.widen⟩

instance instAddLaneLayer : Add LaneLayer := ⟨LaneLayer.composed⟩

@[simp] theorem LaneLayer.composed_apply (a b : LaneLayer) :
    a + b = LaneLayer.composed a b := rfl

/-- **Equality of lane-layer counts is fieldwise equality.** -/
theorem LaneLayer.eq_of_fields {a b : LaneLayer} (h1 : a.laneMove = b.laneMove)
    (h2 : a.widen = b.widen) : a = b := by
  cases a
  cases b
  cases h1; cases h2
  rfl

theorem LaneLayer.composed_comm (a b : LaneLayer) : a + b = b + a := by
  cases a
  cases b
  simp [LaneLayer.composed, Nat.add_comm]

theorem LaneLayer.composed_assoc (a b c : LaneLayer) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [LaneLayer.composed, Nat.add_assoc]

/-- `n` dependent copies of a lane-layer record. Temporal and spatial O(1): two `Nat`
multiplications. -/
def LaneLayer.rep (n : Nat) (a : LaneLayer) : LaneLayer :=
  ⟨n * a.laneMove, n * a.widen⟩

@[simp] theorem LaneLayer.rep_zero (a : LaneLayer) : LaneLayer.rep 0 a = LaneLayer.zero := by
  simp [LaneLayer.rep, LaneLayer.zero]

theorem LaneLayer.rep_succ (n : Nat) (a : LaneLayer) :
    LaneLayer.rep (n + 1) a = a + LaneLayer.rep n a := by
  cases a
  simp [LaneLayer.rep, LaneLayer.composed, Nat.succ_mul, Nat.add_comm]

/-- **The composition law over an iterated construction, lane layer.** -/
theorem LaneLayer.rep_add (n m : Nat) (a : LaneLayer) :
    LaneLayer.rep (n + m) a = LaneLayer.rep n a + LaneLayer.rep m a := by
  cases a
  simp [LaneLayer.rep, LaneLayer.composed, Nat.add_mul]

theorem LaneLayer.rep_composed (n : Nat) (a b : LaneLayer) :
    LaneLayer.rep n (a + b) = LaneLayer.rep n a + LaneLayer.rep n b := by
  cases a
  cases b
  simp [LaneLayer.rep, LaneLayer.composed, Nat.mul_add]

/-! ## 2. The Keccak-f[1600] bodies and their cost records

### 2.1 The word layer

The canonical bodies are `Crypto.Keccak.Round` over `Vector (BitVec 64) 25`, whose lane index is
`x + 5 * y` (FIPS PUB 202 section 3.2). Each operation count below is read off the definition body
displayed in the theorem next to it; the display is `rfl`-checked in this module. A count is a reading
of a displayed term - nothing in Lean counts the operations of a term - which is AR-15's residual item
1, restated at the Keccak bodies in the receipt.

### 2.2 The lane layer

Two kinds of lane-wise operation occur and neither has a `RoundCost` field:

- **`laneMove`.** `pi` (`Round.lean:42-46`) is the one body that *relocates* lane values: output lane
  `i = x + 5y` is `lane s (((x + 3y) mod 5) + 5x)`. `piIndexNat` below is that source index as a
  function of the destination index; it has exactly one fixed point (lane 0), so 24 of the 25 output
  lanes relocate a value, and each relocation is counted as one `laneMove`. A read or a write that
  keeps the lane's index is element traffic, which the measure does not see (no read field; the same
  blindness `CostBridge.loads_invisible_to_order` exhibits for the registered schedule). `theta`,
  `rho`, `chi` and `iota` keep every output lane at its input index and contribute 0.
- **`widen`.** `xorBlock` (`Sponge.lean:55-60`) builds each lane from eight bytes as
  `(block.getD (8i + j) 0).setWidth 64 <<< (8j)`. The `setWidth 64` is an 8-to-64-bit extension; it
  is not one of the eleven fields (not a ring operation, not a shift, not a bitwise op), so it is
  counted in its own quantity: 25 lanes times 8 byte positions = 200 per `xorBlock` call. -/

/-- **The canonical rotation.** `x.rotateLeft n` is `x <<< (n % 64) ||| x >>> (64 - n % 64)`
(`Init/Data/BitVec/Basic.lean:596-610`): two shifts and one disjunction, exactly AR-15's `ROTR`
inventory. -/
theorem rotateLeft_body_ops (x : Lane) (n : Nat) :
    x.rotateLeft n = x <<< (n % 64) ||| x >>> (64 - n % 64) := rfl

/-- The `theta` column parity `C[x] = lane x ^^^ lane (x+5) ^^^ lane (x+10) ^^^ lane (x+15) ^^^
lane (x+20)`, public transcription of the private `column` (`Round.lean:27-28`); the private body and
this one are definitionally equal, checked by the `rfl` in `theta_body_ops`. -/
def columnPub (s : State) (x : Nat) : Lane :=
  lane s x ^^^ lane s (x + 5) ^^^ lane s (x + 10) ^^^ lane s (x + 15) ^^^ lane s (x + 20)

/-- The column parity is four XOR nodes. -/
theorem columnPub_body_ops (s : State) (x : Nat) :
    columnPub s x =
      lane s x ^^^ lane s (x + 5) ^^^ lane s (x + 10) ^^^ lane s (x + 15) ^^^ lane s (x + 20) :=
  rfl

/-- **The `theta` body.** Per output lane: two `column` applications (8 XOR nodes), one `rotateLeft`,
and two XOR nodes (the `D` disjunction and the state application). -/
theorem theta_body_ops (s : State) :
    theta s = Vector.ofFn (fun i : Fin 25 =>
      let x := i.val % 5
      s.get i ^^^ (columnPub s ((x + 4) % 5) ^^^ (columnPub s ((x + 1) % 5)).rotateLeft 1)) :=
  rfl

/-- **The `rho` body.** Per output lane: one `rotateLeft` by its offset `ROT[i]`. -/
theorem rho_body_ops (s : State) :
    rho s = Vector.ofFn (fun i : Fin 25 => (s.get i).rotateLeft (ROT.get i)) := rfl

/-- The source lane index of `pi` as a function of the destination lane index: output lane
`i = x + 5y` reads `(((x + 3y) mod 5) + 5x)`. -/
def piIndexNat (k : Nat) : Nat := ((k % 5 + 3 * (k / 5)) % 5) + 5 * (k % 5)

/-- **The `pi` body**, indexed by the source map: every output lane is one lane relocation. -/
theorem pi_body_ops (s : State) :
    pi s = Vector.ofFn (fun i : Fin 25 => lane s (piIndexNat i.val)) := rfl

/-- Lane 0 is the fixed point of `pi`. -/
theorem piIndexNat_zero : piIndexNat 0 = 0 := by decide

/-- **`pi` fixes lane 0 and only lane 0**, decided over the 25 lane indices. -/
theorem piIndexNat_fixed : ∀ i : Fin 25, piIndexNat i.val = i.val → i = 0 := by decide

/-- **`pi` is injective**, decided over the 25 lane indices; with the finite carrier this is a
permutation of the 25 lanes. -/
theorem piIndexNat_injective :
    ∀ i j : Fin 25, piIndexNat i.val = piIndexNat j.val → i = j := by decide

/-- **Exactly 24 of the 25 lanes change index under `pi`**, counted over the explicit index list. -/
theorem piIndexNat_moved : ((List.range 25).filter (fun k => piIndexNat k ≠ k)).length = 24 := by
  decide

/-- The lane moves of one `pi`. -/
def piMoves : Nat := 24

/-- The counted moves equal the computed moved-lane count. -/
theorem piMoves_eq : piMoves = ((List.range 25).filter (fun k => piIndexNat k ≠ k)).length := by
  rw [piMoves, piIndexNat_moved]

/-- **The `chi` body.** Per output lane: one `~~~`, one `&&&`, one outer `^^^`. -/
theorem chi_body_ops (s : State) :
    chi s = Vector.ofFn (fun i : Fin 25 =>
      let x := i.val % 5
      let y := i.val / 5
      s.get i ^^^
        (~~~ (lane s (((x + 1) % 5) + 5 * y)) &&& lane s (((x + 2) % 5) + 5 * y))) := rfl

/-- **The `iota` body.** One XOR at lane 0 and one same-index write (traffic, invisible). -/
theorem iota_body_ops (n : Fin 24) (s : State) :
    iota n s = s.set 0 (s.get ⟨0, by decide⟩ ^^^ RC.get n) := rfl

/-- **The round body**: `iota[n] (chi (pi (rho (theta s))))`. -/
theorem round_body_ops (n : Fin 24) (s : State) :
    round n s = iota n (chi (pi (rho (theta s)))) := rfl

/-- **The permutation body**: the canonical 24-round left fold. -/
theorem permute_body_ops (s : State) :
    permute s = (List.finRange 24).foldl (fun acc n => round n acc) s :=
  permute_eq_foldl_round s

/-! ## 3. The cost records of the Keccak bodies -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`
(`AnyRing.lean:109-110`): five multiplications and two additions. -/
def ringMulCost (σ : R) : RoundCost σ :=
  { mul := 5, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- Unfolding the canonical multiplication exhibits the body whose operations `ringMulCost` names. -/
theorem ringMul_body_ops (σ : R) (z w : RingCell R) :
    ringMul σ z w = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- **The canonical channel read** `z.1 + r*z.2` (`ChannelsAnyRing.lean:139`): one multiplication,
one addition, one application counted in `channel`. -/
def ringUpperCost (σ : R) : RoundCost σ :=
  { mul := 1, add := 1, sub := 0, neg := 0, cast := 0, channel := 1,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

theorem ringUpper_body_ops (r : R) (z : RingCell R) : ringUpper r z = z.1 + r * z.2 := rfl

/-- **The C5 rotation reading**: one `ringMul` plus one `ringUpper`, `⟨6, 3, 0, 0, 0, 1⟩`. Its
*count* is a fact about the two ring bodies; its identification with a lane rotation is candidate C5
(`ops-map.md:167-180`) at grade `model`, carried as the named premise `RotRead64` in section 6. -/
def rotReadCost (σ : R) : RoundCost σ := ringMulCost σ + ringUpperCost σ

/-- The rotation reading's closed count. -/
theorem rotReadCost_fields (σ : R) :
    rotReadCost σ =
      { mul := 6, add := 3, sub := 0, neg := 0, cast := 0, channel := 1,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } := by
  simp [rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed]

/-- **One rotation in the word layer**: two shifts and one disjunction, AR-15's `rotrCost`. -/
def rotlCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 2, bitOr := 1 }

theorem rotlCost_fields (σ : R) :
    rotlCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 2, bitOr := 1 } := rfl

/-- **One column parity**: four XOR nodes. -/
def columnCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 4, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

theorem columnCost_fields (σ : R) :
    columnCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 4, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } := rfl

/-- **The `theta` cost.** 25 lanes, each with two `column` applications (50 calls of 4 XOR nodes),
one rotation (25 `rotateLeft` calls), and two XOR nodes (25 `D` disjunctions and 25 state
applications). Temporal and spatial O(1): a closed term of three record compositions. Counted in: the
eleven `RoundCost` fields. Derivation: `rep 50 columnCost + rep 25 rotlCost + 50 XOR nodes`. -/
def thetaCost (σ : R) : RoundCost σ :=
  RoundCost.rep σ 50 (columnCost σ) + RoundCost.rep σ 25 (rotlCost σ) +
    ({ mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
       bitXor := 50, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } : RoundCost σ)

/-- **The `theta` closed count**: 250 XORs, 50 shifts, 25 disjunctions. -/
theorem thetaCost_fields (σ : R) :
    thetaCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 250, bitAnd := 0, bitNot := 0, shift := 50, bitOr := 25 } := by
  simp [thetaCost, RoundCost.rep, RoundCost.composed, columnCost, rotlCost]

/-- The multiplicity reading of `thetaCost`, kernel-checked: 50 column calls of 4 XORs, 25 rotations
of 2 shifts, and two XOR nodes per lane. -/
theorem thetaCost_multiplicities (σ : R) :
    (columnCost σ).bitXor = 4 ∧ (rotlCost σ).shift = 2 ∧
      (thetaCost σ).bitXor = 50 * 4 + 2 * 25 ∧ (thetaCost σ).shift = 25 * 2 ∧
      (thetaCost σ).bitOr = 25 := by
  rw [thetaCost_fields, columnCost_fields, rotlCost_fields]
  simp

/-- **The deduplicated `theta`**: with each column parity computed once per `x` (5 calls of 4 XOR
nodes) and each distinct rotation applied once (5 calls), the same step is 50 XORs, 10 shifts and 5
disjunctions. A body that hoists the two `column` calls would cost this; the canonical body does
not. -/
def thetaDistinctCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 50, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 5 }

/-- **The deduplication gap of `theta`**: 200 XORs, 40 shifts, 20 disjunctions, and therefore 20
rotation calls, separate the canonical body from the hoisted one. -/
theorem thetaCost_gap (σ : R) :
    (thetaCost σ).bitXor = (thetaDistinctCost σ).bitXor + 200 ∧
      (thetaCost σ).shift = (thetaDistinctCost σ).shift + 40 ∧
      (thetaCost σ).bitOr = (thetaDistinctCost σ).bitOr + 20 := by
  rw [thetaCost_fields, thetaDistinctCost]
  simp

/-- **The `rho` cost**: 25 rotations. -/
def rhoCost (σ : R) : RoundCost σ := RoundCost.rep σ 25 (rotlCost σ)

theorem rhoCost_fields (σ : R) :
    rhoCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 50, bitOr := 25 } := by
  simp [rhoCost, RoundCost.rep, rotlCost]

/-- **The `pi` cost in the eleven fields is zero**: the body applies no value operation. Its 24
lane moves are counted in `LaneLayer`. -/
def piCost (σ : R) : RoundCost σ := RoundCost.zero σ

theorem piCost_fields (σ : R) : piCost σ = RoundCost.zero σ := rfl

/-- **The `chi` cost**: 25 NOTs, 25 ANDs, 25 XORs. -/
def chiCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 25, bitAnd := 25, bitNot := 25, shift := 0, bitOr := 0 }

theorem chiCost_fields (σ : R) :
    chiCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 25, bitAnd := 25, bitNot := 25, shift := 0, bitOr := 0 } := rfl

/-- **The `iota` cost**: one XOR. -/
def iotaCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 1, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

theorem iotaCost_fields (σ : R) :
    iotaCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 1, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } := rfl

/-- **The word layer of one round**: `theta + rho + pi + chi + iota`; 276 XORs, 25 ANDs, 25 NOTs,
100 shifts, 50 disjunctions, and no ring operation at all. -/
def roundWordCost (σ : R) : RoundCost σ :=
  thetaCost σ + rhoCost σ + piCost σ + chiCost σ + iotaCost σ

theorem roundWordCost_fields (σ : R) :
    roundWordCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 276, bitAnd := 25, bitNot := 25, shift := 100, bitOr := 50 } := by
  simp [roundWordCost, thetaCost, rhoCost, piCost, chiCost, iotaCost, RoundCost.rep,
    RoundCost.composed, RoundCost.zero, columnCost, rotlCost]

/-- **The word layer of a Keccak round contains no ring operation**: all six ring fields are zero,
by construction of the five step bodies. -/
theorem roundWordCost_ring_zero (σ : R) :
    (roundWordCost σ).mul = 0 ∧ (roundWordCost σ).add = 0 ∧ (roundWordCost σ).sub = 0 ∧
      (roundWordCost σ).neg = 0 ∧ (roundWordCost σ).cast = 0 ∧
      (roundWordCost σ).channel = 0 := by
  rw [roundWordCost_fields]
  simp

/-- **One round under AR-15's convention**: the word bodies above plus the C5 ring reading of each
of the round's 50 rotations, `rep 50 rotReadCost`. The ring layer's identification is `model`
(`RotRead64`); the word layer is unconditional. -/
def roundCost (σ : R) : RoundCost σ := roundWordCost σ + RoundCost.rep σ 50 (rotReadCost σ)

/-- **The round closed count**: 300 ring multiplications, 150 ring additions, 50 channel reads, 276
XORs, 25 ANDs, 25 NOTs, 100 shifts and 50 disjunctions. -/
theorem roundCost_fields (σ : R) :
    roundCost σ =
      { mul := 300, add := 150, sub := 0, neg := 0, cast := 0, channel := 50,
        bitXor := 276, bitAnd := 25, bitNot := 25, shift := 100, bitOr := 50 } := by
  simp [roundCost, roundWordCost, thetaCost, rhoCost, piCost, chiCost, iotaCost, rotReadCost,
    ringMulCost, ringUpperCost, RoundCost.rep, RoundCost.composed, RoundCost.zero, columnCost,
    rotlCost]

theorem roundCost_body_only (σ : R) :
    roundCost σ = roundWordCost σ + RoundCost.rep σ 50 (rotReadCost σ) := rfl

/-- **The round's ring fields come only from the C5 rotation layer**: 50 rotations times
`⟨6, 3, 1⟩`. -/
theorem roundCost_ring_fields (σ : R) :
    (roundCost σ).mul = 300 ∧ (roundCost σ).add = 150 ∧ (roundCost σ).channel = 50 ∧
      (roundCost σ).sub = 0 ∧ (roundCost σ).neg = 0 ∧ (roundCost σ).cast = 0 := by
  rw [roundCost_fields]
  simp

/-- The lane layer of one round: the 24 moves of `pi`, no widening. -/
def roundLayer : LaneLayer := ⟨piMoves, 0⟩

theorem roundLayer_fields : roundLayer = ⟨24, 0⟩ := rfl

/-- The round's lane-move count is the computed moved-lane count of `piIndexNat`. -/
theorem roundLayer_laneMove : roundLayer.laneMove =
    ((List.range 25).filter (fun k => piIndexNat k ≠ k)).length := by
  simp [roundLayer, piMoves_eq]

/-- **The word layer of the permutation**: 24 rounds. -/
def permuteWordCost (σ : R) : RoundCost σ := RoundCost.rep σ 24 (roundWordCost σ)

theorem permuteWordCost_fields (σ : R) :
    permuteWordCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 6624, bitAnd := 600, bitNot := 600, shift := 2400, bitOr := 1200 } := by
  simp [permuteWordCost, roundWordCost, thetaCost, rhoCost, piCost, chiCost, iotaCost,
    RoundCost.rep, RoundCost.composed, RoundCost.zero, columnCost, rotlCost]

/-- **The permutation cost, AR-15 convention**: 24 rounds through the iteration operator. Temporal
and spatial O(1): one `RoundCost.rep`, eleven `Nat` multiplications; the growth is in the counted
quantity. Counted in: the eleven `RoundCost` fields. Derivation: `rep_add` telescopes the fold, and
`permute_body_ops` fixes the body as exactly 24 rounds. -/
def permuteCost (σ : R) : RoundCost σ := RoundCost.rep σ 24 (roundCost σ)

/-- **The permutation closed count**: 7200 ring multiplications, 3600 ring additions, 1200 channel
reads, 6624 XORs, 600 ANDs, 600 NOTs, 2400 shifts and 1200 disjunctions. -/
theorem permuteCost_fields (σ : R) :
    permuteCost σ =
      { mul := 7200, add := 3600, sub := 0, neg := 0, cast := 0, channel := 1200,
        bitXor := 6624, bitAnd := 600, bitNot := 600, shift := 2400, bitOr := 1200 } := by
  simp [permuteCost, roundCost, roundWordCost, thetaCost, rhoCost, piCost, chiCost, iotaCost,
    rotReadCost, ringMulCost, ringUpperCost, RoundCost.rep, RoundCost.composed, RoundCost.zero,
    columnCost, rotlCost]

/-- **The permutation's lane layer**: 24 rounds times 24 moves = 576 lane moves; no widening. -/
def permuteLayer : LaneLayer := LaneLayer.rep 24 roundLayer

theorem permuteLayer_fields : permuteLayer = ⟨576, 0⟩ := rfl

/-- Additivity of the permutation figure: the 24-round fold splits at any point. -/
theorem permuteCost_split (σ : R) :
    permuteCost σ = RoundCost.rep σ 12 (roundCost σ) + RoundCost.rep σ 12 (roundCost σ) := by
  simpa [permuteCost] using RoundCost.rep_add σ 12 12 (roundCost σ)

/-! ## 4. The sponge absorb step

`absorb rate domain m` (`Sponge.lean:85-87`) folds `fun s block => permute (xorBlock rate s block)`
over the padded message's rate-byte blocks from the zero state. The price of one step is therefore
one `xorBlock` call plus one `permute`; the count is the body's, independent of `rate`, because the
`xorBlock` body folds the fixed list `List.range 8` over the fixed 25 lanes. The message-block count
of a padded message is a separate question and is not priced here. -/

/-- **The `xorBlock` cost**: per lane, 8 fold steps (8 XORs and 8 shift-and-position operations) plus
one state XOR: 225 XORs, 200 shifts, 200 byte widenings. -/
def xorBlockCost (σ : R) (_rate : Nat) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 225, bitAnd := 0, bitNot := 0, shift := 200, bitOr := 0 }

theorem xorBlockCost_fields (σ : R) (rate : Nat) :
    xorBlockCost σ rate =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 225, bitAnd := 0, bitNot := 0, shift := 200, bitOr := 0 } := rfl

/-- The `xorBlock` multiplicity reading, kernel-checked: 25 lanes times 8 byte positions, plus one
state XOR per lane. -/
theorem xorBlockCost_multiplicities (σ : R) (rate : Nat) :
    (xorBlockCost σ rate).bitXor = 25 * (8 + 1) ∧ (xorBlockCost σ rate).shift = 25 * 8 :=
  ⟨rfl, rfl⟩

/-- The body is rate-independent: the rate changes the block vector's length, not the loop nested in
the lane constructor. -/
theorem xorBlockCost_rate_independent (σ : R) (r r' : Nat) :
    xorBlockCost σ r = xorBlockCost σ r' := rfl

/-- **The `xorBlock` lane layer**: 25 lanes times 8 byte positions = 200 widenings. -/
def xorBlockLayer (_rate : Nat) : LaneLayer := ⟨0, 200⟩

theorem xorBlockLayer_fields (rate : Nat) : xorBlockLayer rate = ⟨0, 200⟩ := rfl

theorem xorBlockLayer_widen (rate : Nat) : (xorBlockLayer rate).widen = 25 * 8 := rfl

/-- **The `xorBlock` body.** Per lane: `s.get i` XOR the eight-byte fold; each fold step shifts a
widened byte into position and XORs it. -/
theorem xorBlock_body_ops (rate : Nat) (s : State) (block : Vector (BitVec 8) rate) :
    xorBlock rate s block = Vector.ofFn (fun i : Fin 25 =>
      s.get i ^^^
        (List.range 8).foldl
          (fun acc j => acc ^^^ ((block.getD (8 * i.val + j) 0).setWidth 64 <<< (8 * j)))
          (0 : BitVec 64)) := rfl

/-- **The absorb fold body**: one `permute` after one `xorBlock` per block, from the zero state. -/
theorem absorb_body_ops (rate : Nat) (domain : BitVec 8) (m : List (BitVec 8)) :
    absorb rate domain m =
      (blocks rate (pad rate domain m)).foldl (fun s block => permute (xorBlock rate s block))
        (Vector.replicate 25 0) := rfl

/-- **The absorb step cost**: `xorBlock` plus `permute`. -/
def absorbStepCost (σ : R) (rate : Nat) : RoundCost σ := xorBlockCost σ rate + permuteCost σ

theorem absorbStepCost_split (σ : R) (rate : Nat) :
    absorbStepCost σ rate = xorBlockCost σ rate + permuteCost σ := rfl

/-- **The absorb-step closed count**: 7200 ring multiplications, 3600 ring additions, 1200 channel
reads, 6849 XORs, 600 ANDs, 600 NOTs, 2600 shifts and 1200 disjunctions. -/
theorem absorbStepCost_fields (σ : R) (rate : Nat) :
    absorbStepCost σ rate =
      { mul := 7200, add := 3600, sub := 0, neg := 0, cast := 0, channel := 1200,
        bitXor := 6849, bitAnd := 600, bitNot := 600, shift := 2600, bitOr := 1200 } := by
  simp [absorbStepCost, permuteCost, roundCost, roundWordCost, thetaCost, rhoCost, piCost, chiCost,
    iotaCost, rotReadCost, ringMulCost, ringUpperCost, xorBlockCost, RoundCost.rep,
    RoundCost.composed, RoundCost.zero, columnCost, rotlCost]

/-- **The absorb-step lane layer**: 576 moves (the permutation) and 200 widenings (the injection). -/
def absorbStepLayer (rate : Nat) : LaneLayer := xorBlockLayer rate + permuteLayer

theorem absorbStepLayer_fields (rate : Nat) : absorbStepLayer rate = ⟨576, 200⟩ := by
  simp [absorbStepLayer, permuteLayer, xorBlockLayer, roundLayer, piMoves, LaneLayer.rep,
    LaneLayer.composed]

/-- The SHA3-256 rate is 136 bytes (`Sponge.lean:29`), so one SHA3-256 absorb block is one
`absorbStepCost` at rate 136. -/
theorem rate_sha3_256_eq : rate_sha3_256 = 136 := rfl

/-- **One SHA3-256 absorb block.** -/
def sha3AbsorbBlockCost (σ : R) : RoundCost σ := absorbStepCost σ rate_sha3_256

theorem sha3AbsorbBlockCost_fields (σ : R) :
    sha3AbsorbBlockCost σ =
      { mul := 7200, add := 3600, sub := 0, neg := 0, cast := 0, channel := 1200,
        bitXor := 6849, bitAnd := 600, bitNot := 600, shift := 2600, bitOr := 1200 } := by
  simpa [sha3AbsorbBlockCost] using absorbStepCost_fields σ rate_sha3_256

/-- **One SHA3-256 absorb block's lane layer.** -/
def sha3AbsorbBlockLayer : LaneLayer := absorbStepLayer rate_sha3_256

theorem sha3AbsorbBlockLayer_fields : sha3AbsorbBlockLayer = ⟨576, 200⟩ := by
  simp [sha3AbsorbBlockLayer, absorbStepLayer, permuteLayer, xorBlockLayer, roundLayer, piMoves,
    LaneLayer.rep, LaneLayer.composed]

/-! ## 5. The comparison table

The SHA-256 row is AR-20's attempt figure (`attemptCost_eq`, kernel-checked in
`work/proofs/AlgebraCostPow.lean` at sha256 a3d62397ffd67d6902c8b28629af42d394df3c55bb176558d8f35
3b1694207e0): one attempt over an 80-byte header is two block transformations, a block is 64 rounds
plus 48 schedule steps plus the eight-word add-back. The per-round and per-schedule-step numerals are
cited from AR-15 (`roundCost_fields`, `schedStepCost_fields`); the composition arithmetic below is
kernel-checked in this module. The two Keccak rows are this module's. -/

/-- AR-20's one-round figure, cited: 36 ring multiplications, 25 additions, 6 channel reads, 8 XORs,
5 ANDs, 1 NOT, 12 shifts, 6 disjunctions. -/
def sha256RoundCost (σ : R) : RoundCost σ :=
  { mul := 36, add := 25, sub := 0, neg := 0, cast := 0, channel := 6,
    bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 }

/-- AR-20's one-schedule-step figure, cited: 24 multiplications, 16 additions, 4 channel reads,
4 XORs, 10 shifts, 4 disjunctions. -/
def sha256SchedStepCost (σ : R) : RoundCost σ :=
  { mul := 24, add := 16, sub := 0, neg := 0, cast := 0, channel := 4,
    bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 }

/-- AR-20's eight-word add-back, cited: eight additions. -/
def sha256AddBackCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 8, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **One SHA-256 block**: 64 rounds, 48 schedule steps, one add-back. -/
def sha256BlockCost (σ : R) : RoundCost σ :=
  RoundCost.rep σ 64 (sha256RoundCost σ) + RoundCost.rep σ 48 (sha256SchedStepCost σ) +
    sha256AddBackCost σ

theorem sha256BlockCost_eq (σ : R) :
    sha256BlockCost σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576 } := by
  simp [sha256BlockCost, sha256RoundCost, sha256SchedStepCost, sha256AddBackCost,
    RoundCost.rep, RoundCost.composed]

/-- **One SHA-256 attempt**: two blocks. -/
def sha256AttemptCost (σ : R) : RoundCost σ := RoundCost.rep σ 2 (sha256BlockCost σ)

theorem sha256AttemptCost_eq (σ : R) :
    sha256AttemptCost σ =
      { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
        bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152 } := by
  rw [sha256AttemptCost, sha256BlockCost_eq]
  simp [RoundCost.rep]

/-- **The comparison table's three vectors, field by field**: SHA-256 one attempt (two blocks, AR-20)
against one Keccak-f[1600] permutation and one SHA3-256 absorb block. Every row is a vector of `Nat`
counts in the same eleven fields; the Keccak rows' lane-layer quantities are recorded separately in
`comparison_layers` because the eleven fields cannot express them. -/
theorem comparison_vectors (σ : R) :
    (sha256AttemptCost σ).vector = (6912, 4752, 0, 0, 0, 1152, 1408, 640, 128, 2496, 1152) ∧
      (permuteCost σ).vector = (7200, 3600, 0, 0, 0, 1200, 6624, 600, 600, 2400, 1200) ∧
      (sha3AbsorbBlockCost σ).vector = (7200, 3600, 0, 0, 0, 1200, 6849, 600, 600, 2600, 1200) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [sha256AttemptCost_eq]
    rfl
  · rw [permuteCost_fields]
    rfl
  · rw [sha3AbsorbBlockCost_fields]
    rfl

/-- **The lane-layer row of the comparison**: the SHA-256 attempt performs no lane move and no byte
widening in this measure; the Keccak permutation performs 576 moves; the SHA3-256 absorb block
performs 576 moves and 200 widenings. -/
theorem comparison_layers :
    (LaneLayer.zero : LaneLayer) = ⟨0, 0⟩ ∧ permuteLayer = ⟨576, 0⟩ ∧
      sha3AbsorbBlockLayer = ⟨576, 200⟩ := ⟨rfl, rfl, sha3AbsorbBlockLayer_fields⟩

/-- The XOR columns of the three rows, stated on their own. -/
theorem comparison_xor (σ : R) :
    (sha256AttemptCost σ).bitXor = 1408 ∧ (permuteCost σ).bitXor = 6624 ∧
      (sha3AbsorbBlockCost σ).bitXor = 6849 := by
  rw [sha256AttemptCost_eq, permuteCost_fields, sha3AbsorbBlockCost_fields]
  simp

/-- **A checked comparison**: in the XOR field the Keccak rows exceed the SHA-256 attempt, and the
absorb block exceeds the permutation by the injection's 225 XORs. -/
theorem comparison_xor_inequality (σ : R) :
    (sha256AttemptCost σ).bitXor < (permuteCost σ).bitXor ∧
      (permuteCost σ).bitXor < (sha3AbsorbBlockCost σ).bitXor ∧
      (sha3AbsorbBlockCost σ).bitXor = (permuteCost σ).bitXor + 225 := by
  rw [sha256AttemptCost_eq, permuteCost_fields, sha3AbsorbBlockCost_fields]
  norm_num

/-! ## 6. The face verdict and the generality boundary

### 6.1 The face verdict

`RoundCost σ` carries a dial index because AR-15's ring fields are read at a dial value. No counted
body of Keccak-f[1600] or of the sponge takes a dial argument, mentions `R`, or branches on one:
the bodies are functions of `BitVec 64` and `BitVec 8` alone. Every Keccak figure in this module is
therefore the *same record* at every dial - the equality below is checked across two arbitrary dial
values - and this is a fact about the counted bodies only. It does not say the trichotomy has no
cost consequences; it says Keccak's canonical bodies contain no ring or dial operation through which
a face could enter. The ring fields of `roundCost` become nonzero only through the C5 rotation
reading, whose identification is `model` and whose statement is the same at every dial for a fixed
root (`channel_read_zero_slot`, AR-15), with the root `r^2 = sigma` existing at the hyperbolic and
parabolic faces and at no elliptic face at widths >= 2. -/

theorem keccak_round_face_blind (σ σ' : R) : (roundCost σ).vector = (roundCost σ').vector := rfl

theorem keccak_permute_face_blind (σ σ' : R) :
    (permuteCost σ).vector = (permuteCost σ').vector := rfl

theorem keccak_absorb_face_blind (σ σ' : R) :
    (sha3AbsorbBlockCost σ).vector = (sha3AbsorbBlockCost σ').vector := rfl

/-- **The rotation reading at width 64, as a named premise.** The channel read of the product
`ringMul 1 (x,0) (2^n, 2^(64-n))` is the left rotation of `x` by `n` bits. Stated, not proved:
candidate C5 at width 64 (`ops-map.md:114`, row K4), grade `model`; it is carried here exactly as
AR-15 carried `RotRead` and is never used as an axiom. -/
def RotRead64 (n : Nat) (x : ZMod (2 ^ 64)) : Prop :=
  ringUpper (1 : ZMod (2 ^ 64))
      (ringMul (1 : ZMod (2 ^ 64)) (x, 0)
        ((2 ^ n : Nat), (2 ^ (64 - n) : Nat))) =
    (((x.val * 2 ^ n) % 2 ^ 64 + x.val / 2 ^ (64 - n) : Nat) : ZMod (2 ^ 64))

/-- The rotation reading's count is unconditional; the premise is deliberately unused. -/
theorem rotRead64_cost_of_reading (σ : R) (n : Nat) (x : ZMod (2 ^ 64)) (_h : RotRead64 n x) :
    (rotReadCost σ).mul = 6 ∧ (rotReadCost σ).add = 3 ∧ (rotReadCost σ).channel = 1 := by
  rw [rotReadCost_fields]
  simp

/-! ### 6.2 The generality boundary

What this datapoint shows about the measure extending beyond SHA-256:

- **The eleven fields carry the whole Keccak permutation.** Every Word operation of the round and
  the sponge injection is one of `bitXor`, `bitAnd`, `bitNot`, `shift`, `bitOr`; the composition and
  iteration laws are the same `RoundCost` laws, and no new field was introduced for them.
- **The ring fields are empty for the canonical bodies.** Unlike SHA-256, whose round performs
  multiplications and additions, the Keccak bodies are bit-level; a two-slot ring emulation of a
  lane XOR is not what the canonical body does and is not counted. The ring layer that appears in
  `roundCost` is the C5 *reading*, the same `model`-grade layer AR-15 applied to SHA-2 rotations;
  without it, all six ring fields of the permutation are zero.
- **Two lane-layer quantities do not map to any field** and are counted in `LaneLayer`:
  `pi`'s lane moves (24 per round) and the sponge injection's byte widenings (200 per block). A
  bitsliced lane operation maps neither to a ring field nor to a bitwise field: a per-bit cost is a
  different measure with its own convention, the same gap AR-10 and AR-15 record for carries. The
  missing object is named: a bit-level record, not another field here.
- **Element traffic, padding, block counts, squeezing, memory and allocation are not covered.**
  Reads (`s.get`, `lane`, `block.getD`, `RC.get`, `ROT.get`) and same-index writes are invisible
  because no field counts them, the same blindness as before. The number of padded blocks of a
  message, and the squeezing phase, are not priced here.
- **The measure is uncalibrated and the figures are counts.** No wall-clock, cycle, joule, watt,
  hash-rate, USD or profitability figure follows from any number in this module, and no hardware
  comparison is made. AGENTS.md:429. -/

end algebra_cost_keccak

/-! ## 7. Axiom cone -/

namespace algebra_cost_keccak

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
#print axioms RingCost.rep_add
#print axioms RingCost.rep_composed
#print axioms ringFields_add
#print axioms ringFields_rep
#print axioms LaneLayer.eq_of_fields
#print axioms LaneLayer.composed_comm
#print axioms LaneLayer.composed_assoc
#print axioms LaneLayer.rep_zero
#print axioms LaneLayer.rep_succ
#print axioms LaneLayer.rep_add
#print axioms LaneLayer.rep_composed
#print axioms rotateLeft_body_ops
#print axioms columnPub_body_ops
#print axioms theta_body_ops
#print axioms rho_body_ops
#print axioms pi_body_ops
#print axioms chi_body_ops
#print axioms iota_body_ops
#print axioms round_body_ops
#print axioms permute_body_ops
#print axioms piIndexNat_zero
#print axioms piIndexNat_fixed
#print axioms piIndexNat_injective
#print axioms piIndexNat_moved
#print axioms piMoves_eq
#print axioms ringMul_body_ops
#print axioms ringUpper_body_ops
#print axioms rotReadCost_fields
#print axioms rotlCost_fields
#print axioms columnCost_fields
#print axioms thetaCost_fields
#print axioms thetaCost_multiplicities
#print axioms thetaCost_gap
#print axioms rhoCost_fields
#print axioms piCost_fields
#print axioms chiCost_fields
#print axioms iotaCost_fields
#print axioms roundWordCost_fields
#print axioms roundWordCost_ring_zero
#print axioms roundCost_fields
#print axioms roundCost_body_only
#print axioms roundCost_ring_fields
#print axioms roundLayer_fields
#print axioms roundLayer_laneMove
#print axioms permuteWordCost_fields
#print axioms permuteCost_fields
#print axioms permuteLayer_fields
#print axioms permuteCost_split
#print axioms xorBlockCost_fields
#print axioms xorBlockCost_multiplicities
#print axioms xorBlockCost_rate_independent
#print axioms xorBlockLayer_fields
#print axioms xorBlockLayer_widen
#print axioms xorBlock_body_ops
#print axioms absorb_body_ops
#print axioms absorbStepCost_split
#print axioms absorbStepCost_fields
#print axioms absorbStepLayer_fields
#print axioms rate_sha3_256_eq
#print axioms sha3AbsorbBlockCost_fields
#print axioms sha3AbsorbBlockLayer_fields
#print axioms sha256BlockCost_eq
#print axioms sha256AttemptCost_eq
#print axioms comparison_vectors
#print axioms comparison_layers
#print axioms comparison_xor
#print axioms comparison_xor_inequality
#print axioms keccak_round_face_blind
#print axioms keccak_permute_face_blind
#print axioms keccak_absorb_face_blind
#print axioms rotRead64_cost_of_reading

end algebra_cost_keccak
