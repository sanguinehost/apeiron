/-
Width-32 slice of the Boolean carrier development: `B_32 = Fin 32 → ZMod 2` with the AR-37 packing
round-trips, the four wiring links, the sigma0/sigma1 ROTR/SHR amounts, the two-input carry step and
the three-stage schedule invariant, each checked at `k = 32`.

`work/proofs/AlgebraBoolCarrier.lean` (AR-37, sha256 9ca8fde7772eef4a0d50ac5219badeaf9e9500e2da8aa1e7d8220008131549d1, read-only in this task) is
the width-generic artifact. `work/proofs` is not an importable module root, so §1–§6 below
re-declare AR-37's development from the same bodies, statement-for-statement, under this namespace;
§7 is the new content and states every result at width 32. Nothing in §7 enumerates the width-32
word space: `decide` appears only on ground literals of bounded size and on the copied small-width
instances (`k ≤ 8`).

Not in scope: the SHA-256 round function as a whole (the round step, the schedule loop, the
compression composition), any executing `Cell<i64>` refinement, and any security, performance or
mining claim. The exact remaining round-function gap on this carrier is §7.4 and the companion `.md`.

No `sorry`, no `admit`, no axiom, no `native_decide`, no raised `maxRecDepth`, no
`maxHeartbeats 0`.
-/
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Data.BitVec
import Mathlib.Tactic
import Lean

set_option autoImplicit false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraBoolCarrier32
open Apeiron.Trichotomy.AnyRing

/-! ## Fail-closed axiom cone

`#audit` is `work/proofs/AlgebraChi.lean`'s command verbatim: it collects each constant's
transitive axioms and throws on anything outside `{propext, Classical.choice, Quot.sound}`, so a
`sorryAx` below is a build failure rather than a line in a log. -/

open Lean Elab Command in
private def auditClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"

/-! ## §1 · The carrier, stated exactly

`B_k = Fin k → ZMod 2` with pointwise operations. `AnyRing` takes an arbitrary commutative ring as
the coefficient ring, so `B_k` is a permitted coefficient ring rather than a new construction, and
`RingCell B_k = B_k × B_k` is then the two-slot algebra over it. This is the adjudication §3
boundary: base-ring choice is "an admissible construction direction, not an established Rust
backend". -/

/-- The Boolean coefficient carrier at width `k`: coordinate `i` is the `i`-th bit, as an element
of `ZMod 2`. All operations are pointwise. -/
abbrev BoolRing (k : ℕ) : Type := Fin k → ZMod 2

/-- The two-slot algebra over the Boolean carrier: `RingCell B_k = B_k × B_k`, the same shape as
`AlgebraCarry`'s `RingCell (ZMod (2^k))` with a different coefficient ring. -/
abbrev BoolCell (k : ℕ) : Type := RingCell (BoolRing k)

/-- `2^k` is positive at every width, so the word ring and its representative map are defined at
every width and not only at `k ≥ 1`. -/
theorem two_pow_pos (k : ℕ) : 0 < 2 ^ k := by
  induction k with
  | zero => norm_num
  | succ m => positivity

/-- `1 < 2^k` at every positive width. -/
theorem one_lt_two_pow {k : ℕ} (hk : 0 < k) : 1 < 2 ^ k := by
  have h := (Nat.pow_le_pow_iff_right (a := 2) (n := 1) (m := k) (by omega)).2 hk
  omega

/-- The Boolean coefficient ring `ZMod 2` read as `Bool`. -/
def f2b (x : ZMod 2) : Bool := decide (x ≠ 0)

/-- A `Bool` read as an element of the Boolean coefficient ring. -/
def b2f (b : Bool) : ZMod 2 := if b then 1 else 0

/-- The two readings are inverse on the two elements of `ZMod 2`. -/
theorem f2b_b2f : ∀ b : Bool, f2b (b2f b) = b := by decide

/-- The two readings are inverse on the two elements of `ZMod 2`. -/
theorem b2f_f2b : ∀ x : ZMod 2, b2f (f2b x) = x := by decide

/-- **On one bit, `b2f` turns exclusive-or into the ring sum.** This is the coefficient-level
content of the XOR row: Boolean addition is bitwise exclusive-or, and `ZMod 2` addition is the
sum `AlgebraChi.ringMul_first_axis` reads. -/
theorem b2f_xor : ∀ a b : Bool, b2f (a ^^ b) = b2f a + b2f b := by decide

/-- **On one bit, `b2f` turns conjunction into the ring product.** -/
theorem b2f_and : ∀ a b : Bool, b2f (a && b) = b2f a * b2f b := by decide

/-- **On one bit, `b2f` turns negation into `1 + ·`.** This is the NOT row: the complement is the
ring element plus one, needing no second slot and no face. -/
theorem b2f_not : ∀ a : Bool, b2f (!a) = 1 + b2f a := by decide

/-! ## §2 · Bit packing and its inverse

`unpack` reads bit `i` of a word as a coefficient; `pack` rebuilds the word from the coordinates,
least significant coordinate first, by recursion on the width. Both composites are proved, so the
relation is an equivalence and not a one-sided embedding. -/

/-- The coordinate read of a `k`-bit word: coordinate `i` is bit `i`, as an element of `ZMod 2`. -/
def unpack {k : ℕ} (x : BitVec k) : BoolRing k := fun i => b2f (x.getLsb i)

/-- The single-bit word of a `Bool`, at any width: bit `0` is `b` and every other bit is zero.
`O(1)`: one `Bool` test. -/
def bitWord (k : ℕ) (b : Bool) : BitVec k := BitVec.ofNat k (if b then 1 else 0)

/-- A zero word has a zero bit at every position, at every width. This is what makes the zero fill
in `wiringShr` a statement about the boundary rather than a convention. -/
theorem getLsbD_zero_word (k : ℕ) (i : ℕ) : (0 : BitVec k).getLsbD i = false := by
  show Nat.testBit 0 i = false
  simp [Nat.testBit]

/-- A `BitVec` of width `k` has a zero bit at every position at or above `k`. -/
theorem getLsbD_ge_width {k : ℕ} (x : BitVec k) (i : ℕ) (h : k ≤ i) : x.getLsbD i = false := by
  rw [BitVec.getLsbD]
  refine Nat.testBit_lt_two_pow ?_
  exact lt_of_lt_of_le x.isLt (Nat.pow_le_pow_right (by norm_num) h)

/-- The bitwise complement reads as the complement of the bit, with the in-range guard made
explicit; `BitVec.getLsbD_not` with the width argument fixed. -/
theorem getLsbD_not_eq {k : ℕ} (x : BitVec k) (i : ℕ) :
    (~~~ x).getLsbD i = (decide (i < k) && ! x.getLsbD i) := BitVec.getLsbD_not

/-- The single-bit word reads back its own bit and is zero everywhere else. -/
theorem bitWord_getLsbD {k : ℕ} (hk : 0 < k) (b : Bool) (i : ℕ) :
    (bitWord k b).getLsbD i = (i = 0 && b) := by
  rcases b with false | true
  · simp [bitWord, Nat.testBit]
  · show Nat.testBit ((1 : ℕ) % 2 ^ k) i = (i = 0 && true)
    rw [Nat.mod_eq_of_lt (one_lt_two_pow hk)]
    rcases i with _ | i
    · rw [Nat.testBit_one_zero]
      rfl
    · rw [Nat.testBit_lt_two_pow (by have h := two_pow_pos (i + 1); omega)]
      rfl

/-- **Packing**, by recursion on the width. Coordinate `0` becomes bit `0`; the remaining `k`
coordinates become the packing of the tail shifted up by one. `O(k)` steps, each `O(1)` on a
fixed-width word; the recursion closes because the width strictly decreases. -/
def pack : (k : ℕ) → BoolRing k → BitVec k
  | 0, _ => 0
  | (k + 1), v =>
      bitWord (k + 1) (f2b (v ⟨0, Nat.succ_pos k⟩))
        ||| (pack k (fun t : Fin k => v t.succ)).shiftLeftZeroExtend 1

/-- **Packing reads back each coordinate, at every width.** -/
theorem pack_getLsbD : ∀ (k : ℕ) (v : BoolRing k) (i : ℕ) (hi : i < k),
    (pack k v).getLsbD i = f2b (v ⟨i, hi⟩) := by
  intro k
  induction k with
  | zero =>
    intro v i hi
    omega
  | succ k ih =>
    intro v i hi
    haveI : (0 : ℕ) < k + 1 := Nat.succ_pos k
    by_cases h0 : i = 0
    · subst i
      show (bitWord (k + 1) (f2b (v ⟨0, Nat.succ_pos k⟩)) |||
          (pack k (fun t : Fin k => v t.succ)).shiftLeftZeroExtend 1).getLsbD 0
          = f2b (v ⟨0, Nat.succ_pos k⟩)
      rw [BitVec.getLsbD_or, BitVec.getLsbD_shiftLeftZeroExtend, bitWord_getLsbD]
      have g0 : (decide (0 = 0)) = true := by simp
      have g1 : (decide (0 < 1)) = true := by simp
      rw [g0, Bool.true_and, g1, Bool.not_true, Bool.false_and]
      simp only [Bool.false_or, Bool.or_false]
      exact Nat.succ_pos k
    · have hge : 1 ≤ i := Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero h0)
      show (bitWord (k + 1) (f2b (v ⟨0, Nat.succ_pos k⟩)) |||
          (pack k (fun t : Fin k => v t.succ)).shiftLeftZeroExtend 1).getLsbD i
          = f2b (v ⟨i, by omega⟩)
      rw [BitVec.getLsbD_or, BitVec.getLsbD_shiftLeftZeroExtend, bitWord_getLsbD]
      have e0'' : (decide (i = 0)) = false := by
        simp only [decide_eq_false_iff_not]
        omega
      have e1' : (decide (i < 1)) = false := by
        simp only [decide_eq_false_iff_not]
        omega
      rw [e0'', Bool.false_and, e1']
      simp only [Bool.not_false, Bool.true_and, Bool.false_or]
      rw [ih (fun t : Fin k => v t.succ) (i - 1) (by omega)]
      have hsub : i - 1 + 1 = i := by omega
      have hval : ((⟨i - 1, by omega⟩ : Fin k).succ : Fin (k + 1)) = ⟨i, hi⟩ := by
        apply Fin.ext
        exact hsub
      rw [hval]
      exact Nat.succ_pos k

/-- **Packing is a left inverse of unpacking: `pack ∘ unpack = id`, at every width.** -/
theorem pack_unpack (k : ℕ) (x : BitVec k) : pack k (unpack x) = x := by
  refine BitVec.eq_of_getLsbD_eq_iff.mpr ?_
  intro i hi
  rw [pack_getLsbD k (unpack x) i hi]
  exact f2b_b2f _

/-- **Unpacking is a left inverse of packing: `unpack ∘ pack = id`, at every width.** -/
theorem unpack_pack (k : ℕ) (v : BoolRing k) : unpack (pack k v) = v := by
  funext i
  have hd : (pack k v).getLsbD i.val = f2b (v i) := pack_getLsbD k v i.val i.isLt
  show b2f ((pack k v).getLsbD i.val) = v i
  rw [hd]
  exact b2f_f2b _

/-- **The word↔bits relation is an equivalence `BitVec k ≃ Fin k → ZMod 2`**, at every width. Its
two composites are `pack_unpack` and `unpack_pack`. This is the exact shape the packing row asks
for: a bijection of carriers, with no claim about a Rust backend. -/
def boolEquiv (k : ℕ) : BitVec k ≃ BoolRing k where
  toFun := unpack
  invFun := pack k
  left_inv := pack_unpack k
  right_inv := unpack_pack k

/-! ## §3 · The first-axis cell, and the XOR / AND / NOT links

`AlgebraChi.ringMul_first_axis` states that on the first axis `ringMul` is the coefficient product
at every `σ`. The statements below are that link transported to the Boolean carrier and composed
with the packing, so each is a checked equality between a two-slot algebra expression and a
`BitVec` operation, at every width. -/

/-- The first-axis cell of a Boolean vector: the vector in slot one, zero in slot two. -/
def bCell {k : ℕ} (v : BoolRing k) : BoolCell k := (v, 0)

/-- The first axis is the vector. -/
theorem bCell_fst {k : ℕ} (v : BoolRing k) : (bCell v).1 = v := rfl

/-- The second axis of a first-axis cell is zero. -/
theorem bCell_snd {k : ℕ} (v : BoolRing k) : (bCell v).2 = 0 := rfl

/-- Cell addition is componentwise, so it stays on the first axis. -/
theorem bCell_add {k : ℕ} (v w : BoolRing k) : bCell v + bCell w = bCell (v + w) := by
  apply Prod.ext <;> simp [bCell]

/-- **On the first axis `ringMul` is the coefficient product, at every face and every width.**
`AlgebraChi.ringMul_first_axis`, over `B_k` instead of `ZMod 2`. -/
theorem ringMul_first_axis_bool {k : ℕ} (σ v w : BoolRing k) :
    ringMul σ (bCell v) (bCell w) = bCell (v * w) := by
  apply Prod.ext <;> simp only [bCell] <;> ring

/-- **A first-axis element times a second-axis element lands in the second slot, at every face.**
`AlgebraCarry.ringMul_second_of_first`, over `B_k`: the mechanism by which a carry-generating
residual would occupy the second axis when it is nonzero. -/
theorem ringMul_second_of_first_bool {k : ℕ} (σ v w : BoolRing k) :
    ringMul σ (bCell v) (0, w) = (0, v * w) := by
  apply Prod.ext <;> simp only [bCell] <;> ring

/-- **The `F2`-coefficient exclusive-or is the coordinate sum of the packed words.** Each coordinate
is decided by `BitVec.getLsbD_xor` on the two `Bool` values and then evaluated as an equation in
`ZMod 2`, so the statement is about the words and not about a naming convention. -/
theorem unpack_add_xor {k : ℕ} (x y : BitVec k) : unpack x + unpack y = unpack (x ^^^ y) := by
  funext i
  show b2f (x.getLsbD i.val) + b2f (y.getLsbD i.val) = b2f ((x ^^^ y).getLsbD i.val)
  simp only [BitVec.getLsbD_xor]
  cases hx : x.getLsbD i.val <;> cases hy : y.getLsbD i.val <;> decide

/-- **The `F2`-coefficient product is the coordinate product of the packed words.** -/
theorem unpack_mul_and {k : ℕ} (x y : BitVec k) : unpack x * unpack y = unpack (x &&& y) := by
  funext i
  show b2f (x.getLsbD i.val) * b2f (y.getLsbD i.val) = b2f ((x &&& y).getLsbD i.val)
  simp only [BitVec.getLsbD_and]
  cases hx : x.getLsbD i.val <;> cases hy : y.getLsbD i.val <;> decide

/-- **The `F2`-coefficient complement is the coordinate `1 + x`.** -/
theorem unpack_add_one_not {k : ℕ} (x : BitVec k) :
    (1 : BoolRing k) + unpack x = unpack (~~~ x) := by
  funext i
  have hi : i.val < k := i.isLt
  show (1 : ZMod 2) + b2f (x.getLsbD i.val) = b2f ((~~~ x).getLsbD i.val)
  rw [getLsbD_not_eq]
  have e : (decide (i.val < k)) = true := by simp [hi]
  rw [e, Bool.true_and]
  cases hx : x.getLsbD i.val <;> decide

/-- **XOR wiring.** Exclusive-or of two words is the sum of their first-axis cells on the Boolean
carrier, at every width. The face is not a parameter: cell addition does not read it. -/
theorem xor_wiring {k : ℕ} (x y : BitVec k) :
    bCell (unpack x) + bCell (unpack y) = bCell (unpack (x ^^^ y)) := by
  rw [bCell_add, unpack_add_xor]

/-- **XOR wiring, read.** The first slot of that cell is the packed exclusive-or. -/
theorem xor_wiring_read {k : ℕ} (x y : BitVec k) :
    (bCell (unpack x) + bCell (unpack y)).1 = unpack (x ^^^ y) := by
  rw [xor_wiring, bCell_fst]

/-- **AND wiring.** Conjunction of two words is one `ringMul` of their first-axis cells, at every
face and every width. This is the AND row's exact statement: no hyperbolic projector is used and
no second slot is read. -/
theorem and_wiring {k : ℕ} (σ : BoolRing k) (x y : BitVec k) :
    ringMul σ (bCell (unpack x)) (bCell (unpack y)) = bCell (unpack (x &&& y)) := by
  rw [ringMul_first_axis_bool, unpack_mul_and]

/-- **AND wiring, read.** The first slot of that product is the packed conjunction. -/
theorem and_wiring_read {k : ℕ} (σ : BoolRing k) (x y : BitVec k) :
    (ringMul σ (bCell (unpack x)) (bCell (unpack y))).1 = unpack (x &&& y) := by
  rw [and_wiring, bCell_fst]

/-- Adding the ring unit keeps a first-axis cell on the first axis. -/
theorem ringOne_add_bCell {k : ℕ} (v : BoolRing k) :
    ringOne + bCell v = bCell ((1 : BoolRing k) + v) := by
  apply Prod.ext <;> simp [ringOne, bCell]

/-- **NOT wiring.** The complement of a word is the ring unit plus its first-axis cell, at every
width. `ringOne = (1, 0)` is `AnyRing.ringOne`, so the additive complement needs neither a face
nor a `ringMul`. -/
theorem not_wiring {k : ℕ} (x : BitVec k) :
    ringOne + bCell (unpack x) = bCell (unpack (~~~ x)) := by
  rw [ringOne_add_bCell, unpack_add_one_not]

/-- **NOT wiring, read.** -/
theorem not_wiring_read {k : ℕ} (x : BitVec k) :
    (ringOne + bCell (unpack x)).1 = unpack (~~~ x) := by
  rw [not_wiring, bCell_fst]

/-- **The packing is not a ring isomorphism carrying modular addition to exclusive-or**: at width
two the coordinate sum of `unpack 3` and `unpack 1` is not the packing of `3 + 1`. This is the
adjudication's line that the word↔bits relation does not carry modular addition to XOR. -/
theorem xor_ne_add_k2 :
    (unpack (3 : BitVec 2) + unpack (1 : BitVec 2)) ≠ unpack ((3 : BitVec 2) + 1) := by decide

/-- **The same separation at width four.** -/
theorem xor_ne_add_k4 :
    (unpack (9 : BitVec 4) + unpack (1 : BitVec 4)) ≠ unpack ((9 : BitVec 4) + 1) := by decide

/-! ## §4 · ROTR and SHR as coordinate wiring

The adjudication records SHR as "Boolean coordinates with zero-fill/truncation" and ROTR as an
index permutation, and states that wiring is admissible when linked. Both are therefore checked
here as equalities between a Boolean-coordinate map and the packed `BitVec` operation, at every
width. -/

/-- **SHR as a coordinate wiring.** Coordinate `i` receives coordinate `i + n` when `i + n` is in
range and zero otherwise: the high coordinates are dropped and the vacated low coordinates are
zero-filled. This is the fill the mutation control of the receipt removes. -/
def wiringShr {k : ℕ} (v : BoolRing k) (n : ℕ) : BoolRing k :=
  fun i => if h : i.val + n < k then v ⟨i.val + n, h⟩ else 0

/-- **SHR wiring, linked.** The coordinate wiring of the packed word is the packing of the
right-shifted word, at every width and every amount. -/
theorem unpack_shr_wiring {k : ℕ} (x : BitVec k) (n : ℕ) :
    wiringShr (unpack x) n = unpack (x >>> n) := by
  funext i
  show (if h : i.val + n < k then unpack x ⟨i.val + n, h⟩ else 0)
      = b2f ((x >>> n).getLsbD i.val)
  rw [BitVec.getLsbD_ushiftRight]
  by_cases h : i.val + n < k
  · rw [dif_pos h]
    have hd : unpack x ⟨i.val + n, h⟩ = b2f (x.getLsbD (i.val + n)) := rfl
    rw [hd, Nat.add_comm]
  · rw [dif_neg h]
    have hge : k ≤ n + i.val := by omega
    rw [getLsbD_ge_width x (n + i.val) hge]
    rfl

/-- **ROTR as a coordinate wiring.** Coordinate `i` receives coordinate `i + n` when `i + n < k`,
and coordinate `i + n - k` otherwise: a cyclic permutation of the coordinates. The second `if` is
unreachable when `n < k` and is present only so the definition is total;
`wiringRotR_second_total` checks that. -/
def wiringRotR {k : ℕ} (n : ℕ) (v : BoolRing k) : BoolRing k :=
  fun i => if h : i.val + n < k then v ⟨i.val + n, h⟩
           else if h2 : i.val + n - k < k then v ⟨i.val + n - k, h2⟩ else 0

/-- **The second branch of `wiringRotR` is in range whenever the first is not, for `n < k`.** -/
theorem wiringRotR_second_total {k : ℕ} (n : ℕ) (_v : BoolRing k) (i : Fin k) (hn : n < k) :
    i.val + n - k < k := by omega

/-- **ROTR wiring, linked.** The coordinate permutation of the packed word is the packing of the
rotated word, at every width and every amount `n < k`. -/
theorem unpack_rotr_wiring {k : ℕ} (n : ℕ) (hn : n < k) (x : BitVec k) :
    wiringRotR n (unpack x) = unpack (x.rotateRight n) := by
  funext i
  have hi : i.val < k := i.isLt
  show (if h : i.val + n < k then unpack x ⟨i.val + n, h⟩
          else if h2 : i.val + n - k < k then unpack x ⟨i.val + n - k, h2⟩ else 0)
      = b2f ((x.rotateRight n).getLsbD i.val)
  have hget : (x.rotateRight n).getLsbD i.val
      = if i.val < k - n then x.getLsbD (n + i.val)
        else if i.val < k then x.getLsbD (i.val - (k - n)) else false := by
    by_cases h1 : i.val < k - n <;>
      rw [BitVec.getLsbD_rotateRight_of_lt hn] <;> simp [cond, h1, hi]
  rw [hget]
  by_cases h1 : i.val < k - n
  · rw [if_pos h1, dif_pos (by omega : i.val + n < k)]
    have hd : unpack x ⟨i.val + n, by omega⟩ = b2f (x.getLsbD (i.val + n)) := rfl
    rw [hd, Nat.add_comm]
  · have h2 : i.val + n - k < k := by omega
    rw [if_neg h1, dif_neg (by omega : ¬ (i.val + n < k)), dif_pos h2]
    have hd : i.val - (k - n) = i.val + n - k := by omega
    rw [hd, if_pos hi]
    rfl

/-- **Carry transport, at every width.** The carry generated by two words is a coordinate wiring
over `Fin (k+1)`: coordinate `0` is zero and coordinate `j+1` is the conjunction at `j`. The extra
coordinate is not decoration: it is the carry *out* of the top bit, which a `k`-bit representation
cannot hold and which `SumStage.hi` of §6 therefore keeps as an integer. -/
def wiringCarry {k : ℕ} (hk : 0 < k) (v : BoolRing k) : BoolRing (k + 1) :=
  fun i => if i.val = 0 then 0
           else v ⟨i.val - 1, by have := i.isLt; omega⟩

set_option linter.unusedVariables false in
/-- **Carry transport, linked, at every width.** The transported carry is `false` at coordinate `0`
and the product of the two input bits at `j - 1` otherwise; the link to the words is
`BitVec.getLsbD_and` on `x &&& y`. -/
theorem wiringCarry_and_coord {k : ℕ} (hk : 0 < k) (x y : BitVec k) (j : Fin (k + 1)) :
    wiringCarry hk (unpack (x &&& y)) j
      = if h : j.val = 0 then (0 : ZMod 2)
        else b2f (x.getLsbD (j.val - 1)) * b2f (y.getLsbD (j.val - 1)) := by
  show (if h : j.val = 0 then (0 : ZMod 2)
          else b2f ((x &&& y).getLsbD (j.val - 1))) = _
  simp only [BitVec.getLsbD_and]
  by_cases h : j.val = 0
  · rw [dif_pos h, dif_pos h]
  · rw [dif_neg h, dif_neg h]
    cases hx : x.getLsbD (j.val - 1) <;> cases hy : y.getLsbD (j.val - 1) <;> decide

/-- **The carry out of the top bit is coordinate `k`, outside the `k`-bit word.** This is the
coordinate that a wrapping `k`-bit addition drops and that `SumStage.hi` keeps as an integer. -/
theorem wiringCarry_top {k : ℕ} (hk : 0 < k) (x y : BitVec k) :
    wiringCarry hk (unpack (x &&& y)) ⟨k, by omega⟩
      = b2f (x.getLsbD (k - 1)) * b2f (y.getLsbD (k - 1)) := by
  rw [wiringCarry_and_coord]
  show (if h : k = 0 then (0 : ZMod 2)
          else b2f (x.getLsbD (k - 1)) * b2f (y.getLsbD (k - 1)))
      = b2f (x.getLsbD (k - 1)) * b2f (y.getLsbD (k - 1))
  rw [dif_neg (by omega)]

/-! ## §5 · The two-input carry step: the modular residual and the integer overflow count are
different objects

`AlgebraCarry.carryCell σ x y = (denoteBV (x ^^^ y), 2 * denoteBV (x &&& y))`: slot two is a
**modular carry-generating residual**, an element of `ZMod (2^k)`. `AlgebraSha256Sched.quadCarry` is
an **integer whole-word overflow count** in `ℕ`. The adjudication §2 requires these to stay
separate — reduction discards the residual's highest bit, and adding residual cells is not the
overflow invariant. The two notions are defined separately below and never identified, and
`residual_ne_overflow` is a checked witness that they differ at every `k ≥ 2`. -/

/-- A `k`-bit word read as an element of the word ring. `AlgebraCarry.denoteBV`. -/
def denoteBV {k : ℕ} (w : BitVec k) : ZMod (2 ^ k) := (w.toNat : ZMod (2 ^ k))

/-- **Wrapping addition of `k`-bit words is addition in `ZMod (2^k)`.** `AlgebraCarry.denoteBV_add`. -/
theorem denoteBV_add {k : ℕ} (x y : BitVec k) : denoteBV (x + y) = denoteBV x + denoteBV y := by
  rw [denoteBV, BitVec.toNat_add, ZMod.natCast_mod]
  exact Nat.cast_add _ _

/-- **The modular carry-generating residual of `x + y`**: slot two of `AlgebraCarry.carryCell`, an
element of `ZMod (2^k)` equal to `2 * denoteBV (x &&& y)`. Reduction can only see it modulo `2^k`. -/
def residual {k : ℕ} (x y : BitVec k) : ZMod (2 ^ k) := 2 * denoteBV (x &&& y)

/-- **The integer whole-word overflow count of `x + y`**: the number of whole `2^k` units the
integer sum carries out. This is `AlgebraSha256Sched.quadCarry`'s notion for one step, an integer,
and it is **not** the residual above. -/
def overflow2 {k : ℕ} (x y : BitVec k) : ℕ := (x.toNat + y.toNat) / 2 ^ k

/-- **The carry cell**, `AlgebraCarry.carryCell` re-declared from the same body: slot one is the
`F2`-linear exclusive-or part, slot two the modular carry-generating residual. The face argument is
unused in the body. -/
def carryCell {k : ℕ} (_σ : ZMod (2 ^ k)) (x y : BitVec k) : RingCell (ZMod (2 ^ k)) :=
  (denoteBV (x ^^^ y), residual x y)

/-- Slot one is the exclusive-or part. -/
theorem carryCell_fst {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 = denoteBV (x ^^^ y) := rfl

/-- Slot two is the residual. -/
theorem carryCell_snd {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).2 = residual x y := rfl

/-- **The two slots of the carry cell add to the exclusive-or part plus the residual, at every
width and every face.** `AlgebraCarry.carryCell_readback`. This is the whole of what the cell
structure gives; it does not identify the residual with the overflow count. -/
theorem carryCell_slots_add {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 + (carryCell σ x y).2 = denoteBV (x ^^^ y) + residual x y := by
  rw [carryCell_fst, carryCell_snd]

/-- **The hyperbolic read of the carry cell is that sum.** `AlgebraCarry.ringUpper_one_add_slots`
composed with the above; choosing `σ = 1` is what makes both slots visible, while `ringUpper 0` sees
slot one only. -/
theorem ringUpper_one_carryCell {k : ℕ} (x y : BitVec k) :
    ringUpper (1 : ZMod (2 ^ k)) (carryCell 1 x y) = denoteBV (x ^^^ y) + residual x y := by
  simp [ringUpper, carryCell]

/-- **Width two: the two slots recompose the modular sum.** Complete over all `2^4` pairs;
`AlgebraCarry.residual_recomposes_sum_two`. -/
theorem residual_recomposes_two : ∀ (x y : BitVec 2),
    (carryCell (0 : ZMod 4) x y).1 + (carryCell (0 : ZMod 4) x y).2 = denoteBV (x + y) := by
  decide

/-- **Width four: the two slots recompose the modular sum.** Complete over all `2^8` pairs;
`AlgebraCarry.residual_recomposes_sum_four`. -/
theorem residual_recomposes_four : ∀ (x y : BitVec 4),
    (carryCell (0 : ZMod 16) x y).1 + (carryCell (0 : ZMod 16) x y).2 = denoteBV (x + y) := by
  decide

/-- **Width eight, one instance**: `255 + 255` in `ZMod 256`, residual `254`;
`AlgebraCarry.residual_recomposes_sum_eight_instance`. A witness, not an enumeration. -/
theorem residual_recomposes_eight_instance :
    carryCell (k := 8) (0 : ZMod 256) 255 255 = (0, 254) ∧
      (carryCell (k := 8) (0 : ZMod 256) 255 255).1
        + (carryCell (k := 8) (0 : ZMod 256) 255 255).2
        = denoteBV ((255 : BitVec 8) + 255) := by
  decide

/-- **Width two: the carry identity** `x + y = (x ⊕ y) + ((x ∧ y) ≪ 1)`;
`AlgebraCarry.carry_identity_two`. -/
theorem carry_identity_two : ∀ (x y : BitVec 2), x + y = (x ^^^ y) + ((x &&& y) <<< 1) := by
  decide

/-- **Width four: the carry identity**; `AlgebraCarry.carry_identity_four`. -/
theorem carry_identity_four : ∀ (x y : BitVec 4), x + y = (x ^^^ y) + ((x &&& y) <<< 1) := by
  decide

/-- **Width eight, one instance.** -/
theorem carry_identity_eight_instance :
    ((255 : BitVec 8) + 255) = ((255 : BitVec 8) ^^^ 255) + ((255 : BitVec 8) &&& 255) <<< 1 := by
  decide

/-- **The augmented carry identity at width four.** Over `k+1` bits nothing is truncated, so the
identity is an exact equality of `BitVec 5` values; the extra top bit of the sum is exactly the
carry out of bit three, which is the coordinate `wiringCarry` isolates. -/
theorem carry_identity_four_augmented : ∀ (x y : BitVec 4),
    x.setWidth 5 + y.setWidth 5 = (x ^^^ y).setWidth 5 + ((x &&& y).setWidth 5 <<< 1) := by
  decide

/-- **The canonical representative of the residual.** The reduction by `2^k` is visible here: this
is what makes "the residual loses its highest bit" a statement rather than a warning. -/
theorem residual_val {k : ℕ} (x y : BitVec k) :
    (residual x y).val = (2 * (x &&& y).toNat) % 2 ^ k := by
  have h : residual x y = ((2 * (x &&& y).toNat : ℕ) : ZMod (2 ^ k)) := by
    unfold residual
    simp [denoteBV, Nat.cast_ofNat, Nat.cast_pow, Nat.cast_mul]
  rw [h, ZMod.val_natCast]

/-- The all-ones word of width `k`. -/
def full (k : ℕ) : BitVec k := ~~~((0 : BitVec k))

/-- The all-ones word has representative `2^k - 1`. -/
theorem full_val (k : ℕ) : (full k).toNat = 2 ^ k - 1 := by simp [full]

/-- The conjunction of two all-ones words is the all-ones word. -/
theorem full_and_full (k : ℕ) : full k &&& full k = full k := by
  refine BitVec.eq_of_getLsbD_eq_iff.mpr ?_
  intro i hi
  simp [full, getLsbD_zero_word, hi]

/-- **Width two: the modular residual and the integer whole-word overflow count are different
quantities.** At `x = y = 2^2 - 1` the residual is `2` in `ZMod 4` while the overflow count is `1`.
Checked, complete over the two named words. -/
theorem residual_ne_overflow_k2 :
    residual (full 2) (full 2) ≠ ((overflow2 (full 2) (full 2) : ℕ) : ZMod 4) := by decide

/-- **Width four: the same separation.** The residual is `14` in `ZMod 16`, the count is `1`. -/
theorem residual_ne_overflow_k4 :
    residual (full 4) (full 4) ≠ ((overflow2 (full 4) (full 4) : ℕ) : ZMod 16) := by decide

/-- **Width eight, one instance: the same separation.** Residual `254` in `ZMod 256`, count `1`. -/
theorem residual_ne_overflow_k8_instance :
    residual (full 8) (full 8) ≠ ((overflow2 (full 8) (full 8) : ℕ) : ZMod 256) := by decide

/-- **The residual's canonical representative is strictly below the modulus, at every width**, so
its value lives in `ZMod (2^k)` and cannot be read as an unbounded integer count. -/
theorem residual_val_lt {k : ℕ} (x y : BitVec k) : (residual x y).val < 2 ^ k := by
  rw [residual_val]
  exact Nat.mod_lt _ (two_pow_pos k)

/-! ## §6 · The three-stage schedule invariant

`a + b + c + d` is three two-input stages: `a+b`, then `+c`, then `+d`. The state carries the
overflow count as an **integer**, separately from the represented `k`-bit word, and the invariant
is that the two together represent the running integer total exactly. That is the invariant the
adjudication §2 requires a sequential construction to preserve; adding residual cells is not it. -/

/-- **The state of the sequential schedule.** `lo` is the represented `k`-bit word; `hi` is the
integer count of whole `2^k` units carried out so far; `n` is the number of width-`k` words fed,
which bounds `hi`. The count is an integer and is **not** a second slot of a `ZMod (2^k)` cell. -/
structure SumStage (k n : ℕ) where
  /-- The represented word: the running total reduced modulo `2^k`. -/
  lo : BitVec k
  /-- The integer whole-word overflow count. -/
  hi : ℕ
  /-- `n` words of width `k` carry out fewer than `n` whole units. -/
  hi_lt : hi < n

/-- **One two-input stage.** Its two inputs are the represented word `s.lo` and the new word `w`; the
state is re-encoded exactly, so the running integer total is preserved and the count is incremented
by the number of whole units this stage carries out. -/
def addStep {k n : ℕ} (s : SumStage k n) (w : BitVec k) : SumStage k (n + 1) where
  lo := s.lo + w
  hi := s.hi + (s.lo.toNat + w.toNat) / 2 ^ k
  hi_lt := by
    have hlo := s.lo.isLt
    have hhi := s.hi_lt
    have hw := w.isLt
    have hq : (s.lo.toNat + w.toNat) / 2 ^ k < 2 := by
      rw [Nat.div_lt_iff_lt_mul (two_pow_pos k)]
      omega
    omega

/-- **Initialization**: the first word, with zero overflow. -/
def startOf {k : ℕ} (a : BitVec k) : SumStage k 1 where
  lo := a
  hi := 0
  hi_lt := Nat.zero_lt_one

/-- **The stage invariant.** `total` is the running integer total of the words fed so far, and the
state's two components represent it exactly: `total = lo.toNat + 2^k * hi`. -/
def SchedInv (k n : ℕ) (total : ℕ) (s : SumStage k n) : Prop :=
  total = s.lo.toNat + 2 ^ k * s.hi

/-- **One stage preserves the invariant, at every width and every stage index.** -/
theorem addStep_inv {k n : ℕ} {total : ℕ} {s : SumStage k n} {w : BitVec k}
    (h : SchedInv k n total s) : SchedInv k (n + 1) (total + w.toNat) (addStep s w) := by
  simp only [SchedInv, addStep]
  rw [BitVec.toNat_add]
  calc total + w.toNat
      = s.lo.toNat + 2 ^ k * s.hi + w.toNat := congrArg (fun t => t + w.toNat) h
    _ = (s.lo.toNat + w.toNat) + 2 ^ k * s.hi := by ring
    _ = ((s.lo.toNat + w.toNat) % 2 ^ k + 2 ^ k * ((s.lo.toNat + w.toNat) / 2 ^ k))
          + 2 ^ k * s.hi := by rw [Nat.mod_add_div]
    _ = _ := by ring

/-- **Stage zero preserves the invariant**: one word, no overflow. -/
theorem startOf_inv {k : ℕ} (a : BitVec k) : SchedInv k 1 a.toNat (startOf a) := by
  simp [SchedInv, startOf]

/-- **Stage one** of the four-word schedule: the sum `a + b`. -/
def sched3_s1 {k : ℕ} (a b : BitVec k) : SumStage k 2 := addStep (startOf a) b

/-- **Stage two**: the sum `a + b + c`. -/
def sched3_s2 {k : ℕ} (a b c : BitVec k) : SumStage k 3 := addStep (sched3_s1 a b) c

/-- **Stage three**: the sum `a + b + c + d`. -/
def sched3_s3 {k : ℕ} (a b c d : BitVec k) : SumStage k 4 := addStep (sched3_s2 a b c) d

/-- **The three-stage schedule** for the SHA-256 message-schedule shape `a + b + c + d`. -/
def sched3 {k : ℕ} (a b c d : BitVec k) : SumStage k 4 := sched3_s3 a b c d

/-- **The invariant holds after stage one.** -/
theorem sched3_s1_inv {k : ℕ} (a b : BitVec k) :
    SchedInv k 2 (a.toNat + b.toNat) (sched3_s1 a b) := addStep_inv (startOf_inv a)

/-- **The invariant holds after stage two.** -/
theorem sched3_s2_inv {k : ℕ} (a b c : BitVec k) :
    SchedInv k 3 (a.toNat + b.toNat + c.toNat) (sched3_s2 a b c) := addStep_inv (sched3_s1_inv a b)

/-- **The three-stage schedule invariant.** After three two-input stages the state represents the
four-word integer total exactly, with the overflow count as a separate integer component. -/
theorem sched3_s3_inv {k : ℕ} (a b c d : BitVec k) :
    SchedInv k 4 (a.toNat + b.toNat + c.toNat + d.toNat) (sched3 a b c d) :=
  addStep_inv (sched3_s2_inv a b c)

/-- **The represented word after the three stages is the wrapping sum of the four inputs.** -/
theorem sched3_s3_lo {k : ℕ} (a b c d : BitVec k) : (sched3 a b c d).lo = a + b + c + d := rfl

set_option linter.unusedVariables false in
/-- **The integer overflow count after the three stages is the whole-word quotient of the four-term
integer total.** This is the component that has no slot in a `k`-bit representation and no slot in
`AlgebraCarry.carryCell`. -/
theorem sched3_s3_hi {k : ℕ} (hk : 1 ≤ k) (a b c d : BitVec k) :
    (sched3 a b c d).hi = (a.toNat + b.toNat + c.toNat + d.toNat) / 2 ^ k := by
  have hinv := sched3_s3_inv a b c d
  simp only [SchedInv] at hinv
  have hz : (0 : ℕ) < 2 ^ k := two_pow_pos k
  have hlt : (sched3 a b c d).lo.toNat < 2 ^ k := (sched3 a b c d).lo.isLt
  have hmod : (a.toNat + b.toNat + c.toNat + d.toNat) % 2 ^ k = (sched3 a b c d).lo.toNat := by
    have e := Nat.add_mul_mod_self_left (sched3 a b c d).lo.toNat (2 ^ k) (sched3 a b c d).hi
    rw [← hinv, Nat.mod_eq_of_lt hlt] at e
    exact e
  have key := Nat.mod_add_div (a.toNat + b.toNat + c.toNat + d.toNat) (2 ^ k)
  rw [hmod] at key
  refine Nat.mul_left_cancel hz ?_
  omega

/-- **The overflow count is bounded by three**, the structural form of
`AlgebraSha256Sched.quadCarry_lt_four` for four width-`k` inputs. -/
theorem sched3_s3_hi_lt {k : ℕ} (a b c d : BitVec k) : (sched3 a b c d).hi < 4 :=
  (sched3 a b c d).hi_lt

/-! ## §7 · Width-32 instances

The carrier at width 32 is `BoolRing 32 = Fin 32 → ZMod 2`; the two-slot algebra is
`BoolCell 32 = BoolRing 32 × BoolRing 32`. Every statement below is the `k := 32` instance of the
same-named width-generic statement in §2–§6 (the AR-37 copy above). -/

/-- The Boolean carrier at width 32. -/
abbrev BoolRing32 : Type := BoolRing 32

/-- The two-slot algebra at width 32. -/
abbrev BoolCell32 : Type := BoolCell 32

/-! ### 7.1 · Packing round-trips and the carrier equivalence at 32 -/

/-- Coordinate `i` of the packing of a width-32 vector is `f2b` of coordinate `i`. -/
theorem pack_getLsbD_32 (v : BoolRing32) (i : Fin 32) :
    (pack 32 v).getLsbD i.val = f2b (v i) := pack_getLsbD 32 v i.val i.isLt

/-- **`pack ∘ unpack = id` at width 32.** -/
theorem pack_unpack_32 (x : BitVec 32) : pack 32 (unpack x) = x := pack_unpack 32 x

/-- **`unpack ∘ pack = id` at width 32.** -/
theorem unpack_pack_32 (v : BoolRing32) : unpack (pack 32 v) = v := unpack_pack 32 v

/-- **The width-32 carrier equivalence `BitVec 32 ≃ BoolRing 32`**, whose two composites are the
round-trips above. -/
def boolEquiv_32 : BitVec 32 ≃ BoolRing32 := boolEquiv 32

/-- The forward map of `boolEquiv_32` is `unpack`. -/
theorem boolEquiv_32_apply (x : BitVec 32) : boolEquiv_32 x = unpack x := rfl

/-- The inverse map of `boolEquiv_32` is `pack 32`. -/
theorem boolEquiv_32_symm_apply (v : BoolRing32) : boolEquiv_32.symm v = pack 32 v := rfl

/-! ### 7.2 · The four wiring links at width 32

XOR as first-axis cell addition, AND as one first-axis `ringMul` at every face, NOT as the ring unit
plus the cell, ROTR and SHR as coordinate wiring. `bCell` is the first-axis embedding. -/

/-- **XOR wiring at 32.** -/
theorem xor_wiring_32 (x y : BitVec 32) :
    bCell (unpack x) + bCell (unpack y) = bCell (unpack (x ^^^ y)) := xor_wiring x y

/-- **XOR wiring at 32, read.** -/
theorem xor_wiring_read_32 (x y : BitVec 32) :
    (bCell (unpack x) + bCell (unpack y)).1 = unpack (x ^^^ y) := xor_wiring_read x y

/-- **AND wiring at 32**, at every face `σ`. -/
theorem and_wiring_32 (σ : BoolRing32) (x y : BitVec 32) :
    ringMul σ (bCell (unpack x)) (bCell (unpack y)) = bCell (unpack (x &&& y)) :=
  and_wiring σ x y

/-- **AND wiring at 32, read.** -/
theorem and_wiring_read_32 (σ : BoolRing32) (x y : BitVec 32) :
    (ringMul σ (bCell (unpack x)) (bCell (unpack y))).1 = unpack (x &&& y) :=
  and_wiring_read σ x y

/-- **NOT wiring at 32.** -/
theorem not_wiring_32 (x : BitVec 32) :
    ringOne + bCell (unpack x) = bCell (unpack (~~~ x)) := not_wiring x

/-- **NOT wiring at 32, read.** -/
theorem not_wiring_read_32 (x : BitVec 32) :
    (ringOne + bCell (unpack x)).1 = unpack (~~~ x) := not_wiring_read x

/-- **ROTR wiring at 32**, every amount `n < 32`. -/
theorem unpack_rotr_wiring_32 (n : ℕ) (hn : n < 32) (x : BitVec 32) :
    wiringRotR n (unpack x) = unpack (x.rotateRight n) := unpack_rotr_wiring (k := 32) n hn x

/-- **SHR wiring at 32**, every amount. -/
theorem unpack_shr_wiring_32 (n : ℕ) (x : BitVec 32) :
    wiringShr (unpack x) n = unpack (x >>> n) := unpack_shr_wiring x n

/-! #### The six sigma0/sigma1 amounts of FIPS 180-4 §4.1.2 at 32

`sigma0 = ROTR 7 ⊕ ROTR 18 ⊕ SHR 3` and `sigma1 = ROTR 17 ⊕ ROTR 19 ⊕ SHR 10` on 32-bit words:
each term is checked as coordinate wiring on the Boolean carrier, at width 32. -/

/-- **sigma0's ROTR 7** as coordinate wiring at 32. -/
theorem wiring_rotr_7_32 (x : BitVec 32) :
    wiringRotR 7 (unpack x) = unpack (x.rotateRight 7) :=
  unpack_rotr_wiring (k := 32) 7 (by norm_num) x

/-- **sigma0's ROTR 18** as coordinate wiring at 32. -/
theorem wiring_rotr_18_32 (x : BitVec 32) :
    wiringRotR 18 (unpack x) = unpack (x.rotateRight 18) :=
  unpack_rotr_wiring (k := 32) 18 (by norm_num) x

/-- **sigma1's ROTR 17** as coordinate wiring at 32. -/
theorem wiring_rotr_17_32 (x : BitVec 32) :
    wiringRotR 17 (unpack x) = unpack (x.rotateRight 17) :=
  unpack_rotr_wiring (k := 32) 17 (by norm_num) x

/-- **sigma1's ROTR 19** as coordinate wiring at 32. -/
theorem wiring_rotr_19_32 (x : BitVec 32) :
    wiringRotR 19 (unpack x) = unpack (x.rotateRight 19) :=
  unpack_rotr_wiring (k := 32) 19 (by norm_num) x

/-- **sigma0's SHR 3** as coordinate wiring at 32. -/
theorem wiring_shr_3_32 (x : BitVec 32) : wiringShr (unpack x) 3 = unpack (x >>> 3) :=
  unpack_shr_wiring x 3

/-- **sigma1's SHR 10** as coordinate wiring at 32. -/
theorem wiring_shr_10_32 (x : BitVec 32) : wiringShr (unpack x) 10 = unpack (x >>> 10) :=
  unpack_shr_wiring x 10

/-- **The checked amount is load-bearing at 32.** On the witness word `1` the ROTR-7 wiring differs
from the packing of the ROTR-8 word, so `wiring_rotr_7_32` is not a naming convention that any
rotation index would satisfy; the `decide` is a ground literal check, not an enumeration. -/
theorem wiring_rotr_7_ne_8_32 :
    wiringRotR 7 (unpack (1 : BitVec 32)) ≠ unpack ((1 : BitVec 32).rotateRight 8) := by
  intro h
  rw [wiring_rotr_7_32 (1 : BitVec 32)] at h
  exact (by decide : (1 : BitVec 32).rotateRight 7 ≠ (1 : BitVec 32).rotateRight 8)
    ((boolEquiv 32).injective h)

/-! ### 7.3 · Two-input carry step and three-stage schedule invariant at 32 -/

/-- **Wrapping 32-bit addition is addition in `ZMod (2^32)`.** -/
theorem denoteBV_add_32 (x y : BitVec 32) : denoteBV (x + y) = denoteBV x + denoteBV y :=
  denoteBV_add x y

/-- **The two slots of the 32-bit carry cell add to the exclusive-or part plus the residual**, at
every face. -/
theorem carryCell_slots_add_32 (σ : ZMod (2 ^ 32)) (x y : BitVec 32) :
    (carryCell σ x y).1 + (carryCell σ x y).2 = denoteBV (x ^^^ y) + residual x y :=
  carryCell_slots_add σ x y

/-- **The `σ = 1` hyperbolic read of the 32-bit carry cell is that sum.** -/
theorem ringUpper_one_carryCell_32 (x y : BitVec 32) :
    ringUpper (1 : ZMod (2 ^ 32)) (carryCell 1 x y) = denoteBV (x ^^^ y) + residual x y :=
  ringUpper_one_carryCell x y

/-- **The modular residual of 32-bit addition has representative `(2 * (x &&& y)) % 2^32`**: the
residual is an element of `ZMod (2^32)`, and the reduction is stated rather than bypassed. -/
theorem residual_val_32 (x y : BitVec 32) :
    (residual x y).val = (2 * (x &&& y).toNat) % 2 ^ 32 := residual_val x y

/-- The residual's representative stays below the 32-bit modulus. -/
theorem residual_val_lt_32 (x y : BitVec 32) : (residual x y).val < 2 ^ 32 :=
  residual_val_lt x y

/-- **The modular residual and the integer whole-word overflow count are different quantities at
width 32.** At `x = y = 2^32 - 1` the residual has representative `2^32 - 2` while the overflow
count is `1`. `residual_val`, `full_and_full` and `full_val` rewrite both sides to numerals; the
word space is never enumerated. -/
theorem residual_ne_overflow_k32 :
    residual (full 32) (full 32) ≠ ((overflow2 (full 32) (full 32) : ℕ) : ZMod (2 ^ 32)) := by
  intro h
  have hv := congrArg ZMod.val h
  have hres : (residual (full 32) (full 32)).val = 2 ^ 32 - 2 := by
    rw [residual_val, full_and_full, full_val]
    norm_num
  have hov : overflow2 (full 32) (full 32) = 1 := by
    unfold overflow2
    rw [full_val]
    norm_num
  rw [hres, hov, ZMod.val_natCast] at hv
  norm_num at hv

/-- **One two-input stage preserves the schedule invariant at width 32**, for every stage index `n`
and running total. -/
theorem addStep_inv_32 {n total : ℕ} {s : SumStage 32 n} {w : BitVec 32}
    (h : SchedInv 32 n total s) : SchedInv 32 (n + 1) (total + w.toNat) (addStep s w) :=
  addStep_inv h

/-- **The three-stage invariant at width 32**: after `a + b`, `+ c`, `+ d` the state represents the
four-word integer total exactly. -/
theorem sched3_s3_inv_32 (a b c d : BitVec 32) :
    SchedInv 32 4 (a.toNat + b.toNat + c.toNat + d.toNat) (sched3 a b c d) :=
  sched3_s3_inv a b c d

/-- **The represented word after three stages at width 32** is the wrapping four-word sum. -/
theorem sched3_s3_lo_32 (a b c d : BitVec 32) : (sched3 a b c d).lo = a + b + c + d :=
  sched3_s3_lo a b c d

/-- **The integer overflow count after three stages at width 32** is the whole-word quotient of the
four-term integer total: no `ZMod (2^32)` slot and no 32-bit word holds it. -/
theorem sched3_s3_hi_32 (a b c d : BitVec 32) :
    (sched3 a b c d).hi = (a.toNat + b.toNat + c.toNat + d.toNat) / 2 ^ 32 :=
  sched3_s3_hi (k := 32) (by norm_num) a b c d

/-- The overflow count after three stages at 32 is bounded by three. -/
theorem sched3_s3_hi_lt_32 (a b c d : BitVec 32) : (sched3 a b c d).hi < 4 :=
  sched3_s3_hi_lt a b c d

/-- **A reachable width-32 instance**: the four all-ones words carry exactly three whole `2^32`
units out, so the integer count component is exercised on a nonzero state. -/
theorem sched3_full_hi_32 :
    (sched3 (full 32) (full 32) (full 32) (full 32)).hi = 3 := by
  rw [sched3_s3_hi_32 (full 32) (full 32) (full 32) (full 32)]
  rw [full_val]
  norm_num

/-! ### 7.4 · Not in this file: the round step

The six amount wires above are terms of `sigma0`/`sigma1`, not the composites: the `zxor3`
composition `unpack (x.rotateRight 7 ^^^ x.rotateRight 18 ^^^ (x >>> 3))` through `unpack_add_xor`,
the Ch/Maj and Sigma0/Sigma1 layers, a `BoolRing 32`-valued round transition composed with
`SchedInv`, and the refinement from the extracted round body are not stated here. The SHA-256 round
step as a whole is not this task and is not claimed. -/

/-! ## §8 · The closed cone -/

#audit AlgebraBoolCarrier32.BoolRing
#audit AlgebraBoolCarrier32.BoolCell
#audit AlgebraBoolCarrier32.two_pow_pos
#audit AlgebraBoolCarrier32.one_lt_two_pow
#audit AlgebraBoolCarrier32.f2b
#audit AlgebraBoolCarrier32.b2f
#audit AlgebraBoolCarrier32.f2b_b2f
#audit AlgebraBoolCarrier32.b2f_f2b
#audit AlgebraBoolCarrier32.b2f_xor
#audit AlgebraBoolCarrier32.b2f_and
#audit AlgebraBoolCarrier32.b2f_not
#audit AlgebraBoolCarrier32.unpack
#audit AlgebraBoolCarrier32.bitWord
#audit AlgebraBoolCarrier32.getLsbD_zero_word
#audit AlgebraBoolCarrier32.getLsbD_ge_width
#audit AlgebraBoolCarrier32.getLsbD_not_eq
#audit AlgebraBoolCarrier32.bitWord_getLsbD
#audit AlgebraBoolCarrier32.pack
#audit AlgebraBoolCarrier32.pack_getLsbD
#audit AlgebraBoolCarrier32.pack_unpack
#audit AlgebraBoolCarrier32.unpack_pack
#audit AlgebraBoolCarrier32.boolEquiv
#audit AlgebraBoolCarrier32.bCell
#audit AlgebraBoolCarrier32.bCell_fst
#audit AlgebraBoolCarrier32.bCell_snd
#audit AlgebraBoolCarrier32.bCell_add
#audit AlgebraBoolCarrier32.ringMul_first_axis_bool
#audit AlgebraBoolCarrier32.ringMul_second_of_first_bool
#audit AlgebraBoolCarrier32.unpack_add_xor
#audit AlgebraBoolCarrier32.unpack_mul_and
#audit AlgebraBoolCarrier32.unpack_add_one_not
#audit AlgebraBoolCarrier32.xor_wiring
#audit AlgebraBoolCarrier32.xor_wiring_read
#audit AlgebraBoolCarrier32.and_wiring
#audit AlgebraBoolCarrier32.and_wiring_read
#audit AlgebraBoolCarrier32.ringOne_add_bCell
#audit AlgebraBoolCarrier32.not_wiring
#audit AlgebraBoolCarrier32.not_wiring_read
#audit AlgebraBoolCarrier32.xor_ne_add_k2
#audit AlgebraBoolCarrier32.xor_ne_add_k4
#audit AlgebraBoolCarrier32.wiringShr
#audit AlgebraBoolCarrier32.unpack_shr_wiring
#audit AlgebraBoolCarrier32.wiringRotR
#audit AlgebraBoolCarrier32.wiringRotR_second_total
#audit AlgebraBoolCarrier32.unpack_rotr_wiring
#audit AlgebraBoolCarrier32.wiringCarry
#audit AlgebraBoolCarrier32.wiringCarry_and_coord
#audit AlgebraBoolCarrier32.wiringCarry_top
#audit AlgebraBoolCarrier32.denoteBV
#audit AlgebraBoolCarrier32.denoteBV_add
#audit AlgebraBoolCarrier32.residual
#audit AlgebraBoolCarrier32.overflow2
#audit AlgebraBoolCarrier32.carryCell
#audit AlgebraBoolCarrier32.carryCell_fst
#audit AlgebraBoolCarrier32.carryCell_snd
#audit AlgebraBoolCarrier32.carryCell_slots_add
#audit AlgebraBoolCarrier32.ringUpper_one_carryCell
#audit AlgebraBoolCarrier32.residual_recomposes_two
#audit AlgebraBoolCarrier32.residual_recomposes_four
#audit AlgebraBoolCarrier32.residual_recomposes_eight_instance
#audit AlgebraBoolCarrier32.carry_identity_two
#audit AlgebraBoolCarrier32.carry_identity_four
#audit AlgebraBoolCarrier32.carry_identity_eight_instance
#audit AlgebraBoolCarrier32.carry_identity_four_augmented
#audit AlgebraBoolCarrier32.residual_val
#audit AlgebraBoolCarrier32.full
#audit AlgebraBoolCarrier32.full_val
#audit AlgebraBoolCarrier32.full_and_full
#audit AlgebraBoolCarrier32.residual_ne_overflow_k2
#audit AlgebraBoolCarrier32.residual_ne_overflow_k4
#audit AlgebraBoolCarrier32.residual_ne_overflow_k8_instance
#audit AlgebraBoolCarrier32.residual_val_lt
#audit AlgebraBoolCarrier32.addStep
#audit AlgebraBoolCarrier32.startOf
#audit AlgebraBoolCarrier32.SchedInv
#audit AlgebraBoolCarrier32.addStep_inv
#audit AlgebraBoolCarrier32.startOf_inv
#audit AlgebraBoolCarrier32.sched3_s1
#audit AlgebraBoolCarrier32.sched3_s2
#audit AlgebraBoolCarrier32.sched3_s3
#audit AlgebraBoolCarrier32.sched3
#audit AlgebraBoolCarrier32.sched3_s1_inv
#audit AlgebraBoolCarrier32.sched3_s2_inv
#audit AlgebraBoolCarrier32.sched3_s3_inv
#audit AlgebraBoolCarrier32.sched3_s3_lo
#audit AlgebraBoolCarrier32.sched3_s3_hi
#audit AlgebraBoolCarrier32.sched3_s3_hi_lt
#audit AlgebraBoolCarrier32.BoolRing32
#audit AlgebraBoolCarrier32.BoolCell32
#audit AlgebraBoolCarrier32.pack_getLsbD_32
#audit AlgebraBoolCarrier32.pack_unpack_32
#audit AlgebraBoolCarrier32.unpack_pack_32
#audit AlgebraBoolCarrier32.boolEquiv_32
#audit AlgebraBoolCarrier32.boolEquiv_32_apply
#audit AlgebraBoolCarrier32.boolEquiv_32_symm_apply
#audit AlgebraBoolCarrier32.xor_wiring_32
#audit AlgebraBoolCarrier32.xor_wiring_read_32
#audit AlgebraBoolCarrier32.and_wiring_32
#audit AlgebraBoolCarrier32.and_wiring_read_32
#audit AlgebraBoolCarrier32.not_wiring_32
#audit AlgebraBoolCarrier32.not_wiring_read_32
#audit AlgebraBoolCarrier32.unpack_rotr_wiring_32
#audit AlgebraBoolCarrier32.unpack_shr_wiring_32
#audit AlgebraBoolCarrier32.wiring_rotr_7_32
#audit AlgebraBoolCarrier32.wiring_rotr_18_32
#audit AlgebraBoolCarrier32.wiring_rotr_17_32
#audit AlgebraBoolCarrier32.wiring_rotr_19_32
#audit AlgebraBoolCarrier32.wiring_shr_3_32
#audit AlgebraBoolCarrier32.wiring_shr_10_32
#audit AlgebraBoolCarrier32.wiring_rotr_7_ne_8_32
#audit AlgebraBoolCarrier32.denoteBV_add_32
#audit AlgebraBoolCarrier32.carryCell_slots_add_32
#audit AlgebraBoolCarrier32.ringUpper_one_carryCell_32
#audit AlgebraBoolCarrier32.residual_val_32
#audit AlgebraBoolCarrier32.residual_val_lt_32
#audit AlgebraBoolCarrier32.residual_ne_overflow_k32
#audit AlgebraBoolCarrier32.addStep_inv_32
#audit AlgebraBoolCarrier32.sched3_s3_inv_32
#audit AlgebraBoolCarrier32.sched3_s3_lo_32
#audit AlgebraBoolCarrier32.sched3_s3_hi_32
#audit AlgebraBoolCarrier32.sched3_s3_hi_lt_32
#audit AlgebraBoolCarrier32.sched3_full_hi_32

#print axioms AlgebraBoolCarrier32.BoolRing
#print axioms AlgebraBoolCarrier32.BoolCell
#print axioms AlgebraBoolCarrier32.two_pow_pos
#print axioms AlgebraBoolCarrier32.one_lt_two_pow
#print axioms AlgebraBoolCarrier32.f2b
#print axioms AlgebraBoolCarrier32.b2f
#print axioms AlgebraBoolCarrier32.f2b_b2f
#print axioms AlgebraBoolCarrier32.b2f_f2b
#print axioms AlgebraBoolCarrier32.b2f_xor
#print axioms AlgebraBoolCarrier32.b2f_and
#print axioms AlgebraBoolCarrier32.b2f_not
#print axioms AlgebraBoolCarrier32.unpack
#print axioms AlgebraBoolCarrier32.bitWord
#print axioms AlgebraBoolCarrier32.getLsbD_zero_word
#print axioms AlgebraBoolCarrier32.getLsbD_ge_width
#print axioms AlgebraBoolCarrier32.getLsbD_not_eq
#print axioms AlgebraBoolCarrier32.bitWord_getLsbD
#print axioms AlgebraBoolCarrier32.pack
#print axioms AlgebraBoolCarrier32.pack_getLsbD
#print axioms AlgebraBoolCarrier32.pack_unpack
#print axioms AlgebraBoolCarrier32.unpack_pack
#print axioms AlgebraBoolCarrier32.boolEquiv
#print axioms AlgebraBoolCarrier32.bCell
#print axioms AlgebraBoolCarrier32.bCell_fst
#print axioms AlgebraBoolCarrier32.bCell_snd
#print axioms AlgebraBoolCarrier32.bCell_add
#print axioms AlgebraBoolCarrier32.ringMul_first_axis_bool
#print axioms AlgebraBoolCarrier32.ringMul_second_of_first_bool
#print axioms AlgebraBoolCarrier32.unpack_add_xor
#print axioms AlgebraBoolCarrier32.unpack_mul_and
#print axioms AlgebraBoolCarrier32.unpack_add_one_not
#print axioms AlgebraBoolCarrier32.xor_wiring
#print axioms AlgebraBoolCarrier32.xor_wiring_read
#print axioms AlgebraBoolCarrier32.and_wiring
#print axioms AlgebraBoolCarrier32.and_wiring_read
#print axioms AlgebraBoolCarrier32.ringOne_add_bCell
#print axioms AlgebraBoolCarrier32.not_wiring
#print axioms AlgebraBoolCarrier32.not_wiring_read
#print axioms AlgebraBoolCarrier32.xor_ne_add_k2
#print axioms AlgebraBoolCarrier32.xor_ne_add_k4
#print axioms AlgebraBoolCarrier32.wiringShr
#print axioms AlgebraBoolCarrier32.unpack_shr_wiring
#print axioms AlgebraBoolCarrier32.wiringRotR
#print axioms AlgebraBoolCarrier32.wiringRotR_second_total
#print axioms AlgebraBoolCarrier32.unpack_rotr_wiring
#print axioms AlgebraBoolCarrier32.wiringCarry
#print axioms AlgebraBoolCarrier32.wiringCarry_and_coord
#print axioms AlgebraBoolCarrier32.wiringCarry_top
#print axioms AlgebraBoolCarrier32.denoteBV
#print axioms AlgebraBoolCarrier32.denoteBV_add
#print axioms AlgebraBoolCarrier32.residual
#print axioms AlgebraBoolCarrier32.overflow2
#print axioms AlgebraBoolCarrier32.carryCell
#print axioms AlgebraBoolCarrier32.carryCell_fst
#print axioms AlgebraBoolCarrier32.carryCell_snd
#print axioms AlgebraBoolCarrier32.carryCell_slots_add
#print axioms AlgebraBoolCarrier32.ringUpper_one_carryCell
#print axioms AlgebraBoolCarrier32.residual_recomposes_two
#print axioms AlgebraBoolCarrier32.residual_recomposes_four
#print axioms AlgebraBoolCarrier32.residual_recomposes_eight_instance
#print axioms AlgebraBoolCarrier32.carry_identity_two
#print axioms AlgebraBoolCarrier32.carry_identity_four
#print axioms AlgebraBoolCarrier32.carry_identity_eight_instance
#print axioms AlgebraBoolCarrier32.carry_identity_four_augmented
#print axioms AlgebraBoolCarrier32.residual_val
#print axioms AlgebraBoolCarrier32.full
#print axioms AlgebraBoolCarrier32.full_val
#print axioms AlgebraBoolCarrier32.full_and_full
#print axioms AlgebraBoolCarrier32.residual_ne_overflow_k2
#print axioms AlgebraBoolCarrier32.residual_ne_overflow_k4
#print axioms AlgebraBoolCarrier32.residual_ne_overflow_k8_instance
#print axioms AlgebraBoolCarrier32.residual_val_lt
#print axioms AlgebraBoolCarrier32.addStep
#print axioms AlgebraBoolCarrier32.startOf
#print axioms AlgebraBoolCarrier32.SchedInv
#print axioms AlgebraBoolCarrier32.addStep_inv
#print axioms AlgebraBoolCarrier32.startOf_inv
#print axioms AlgebraBoolCarrier32.sched3_s1
#print axioms AlgebraBoolCarrier32.sched3_s2
#print axioms AlgebraBoolCarrier32.sched3_s3
#print axioms AlgebraBoolCarrier32.sched3
#print axioms AlgebraBoolCarrier32.sched3_s1_inv
#print axioms AlgebraBoolCarrier32.sched3_s2_inv
#print axioms AlgebraBoolCarrier32.sched3_s3_inv
#print axioms AlgebraBoolCarrier32.sched3_s3_lo
#print axioms AlgebraBoolCarrier32.sched3_s3_hi
#print axioms AlgebraBoolCarrier32.sched3_s3_hi_lt
#print axioms AlgebraBoolCarrier32.BoolRing32
#print axioms AlgebraBoolCarrier32.BoolCell32
#print axioms AlgebraBoolCarrier32.pack_getLsbD_32
#print axioms AlgebraBoolCarrier32.pack_unpack_32
#print axioms AlgebraBoolCarrier32.unpack_pack_32
#print axioms AlgebraBoolCarrier32.boolEquiv_32
#print axioms AlgebraBoolCarrier32.boolEquiv_32_apply
#print axioms AlgebraBoolCarrier32.boolEquiv_32_symm_apply
#print axioms AlgebraBoolCarrier32.xor_wiring_32
#print axioms AlgebraBoolCarrier32.xor_wiring_read_32
#print axioms AlgebraBoolCarrier32.and_wiring_32
#print axioms AlgebraBoolCarrier32.and_wiring_read_32
#print axioms AlgebraBoolCarrier32.not_wiring_32
#print axioms AlgebraBoolCarrier32.not_wiring_read_32
#print axioms AlgebraBoolCarrier32.unpack_rotr_wiring_32
#print axioms AlgebraBoolCarrier32.unpack_shr_wiring_32
#print axioms AlgebraBoolCarrier32.wiring_rotr_7_32
#print axioms AlgebraBoolCarrier32.wiring_rotr_18_32
#print axioms AlgebraBoolCarrier32.wiring_rotr_17_32
#print axioms AlgebraBoolCarrier32.wiring_rotr_19_32
#print axioms AlgebraBoolCarrier32.wiring_shr_3_32
#print axioms AlgebraBoolCarrier32.wiring_shr_10_32
#print axioms AlgebraBoolCarrier32.wiring_rotr_7_ne_8_32
#print axioms AlgebraBoolCarrier32.denoteBV_add_32
#print axioms AlgebraBoolCarrier32.carryCell_slots_add_32
#print axioms AlgebraBoolCarrier32.ringUpper_one_carryCell_32
#print axioms AlgebraBoolCarrier32.residual_val_32
#print axioms AlgebraBoolCarrier32.residual_val_lt_32
#print axioms AlgebraBoolCarrier32.residual_ne_overflow_k32
#print axioms AlgebraBoolCarrier32.addStep_inv_32
#print axioms AlgebraBoolCarrier32.sched3_s3_inv_32
#print axioms AlgebraBoolCarrier32.sched3_s3_lo_32
#print axioms AlgebraBoolCarrier32.sched3_s3_hi_32
#print axioms AlgebraBoolCarrier32.sched3_s3_hi_lt_32
#print axioms AlgebraBoolCarrier32.sched3_full_hi_32

end AlgebraBoolCarrier32
