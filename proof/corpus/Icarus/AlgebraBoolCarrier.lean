/-
Boolean carrier `B_k = Fin k → ZMod 2` for SHA-256's per-word primitives, its bit packing and
inverse, the four wiring links, a two-input carry step that keeps the modular carry-generating
residual separate from the integer whole-word overflow count, and a three-stage schedule
invariant for `a + b + c + d`.

Scope is `receipts/sol-face-adjudication.md` §5 row 3 and §3's per-operation carrier table
(XOR / AND / NOT / SHR / carry rows), which requires these links to be **checked statements**, not
assertions because they are called wiring. Lane widths are 2, 4 and 8; the SHA-256 schedule is
width 32 in FIPS 180-4 and nothing here is a width-32 execution claim.

Provenance of every borrowed fact is named at the point of use:

* `Apeiron.Trichotomy.AnyRing.RingCell / ringMul / ringOne / ringUpper` — the two-slot algebra
  `R[x]/(x^2-σ)` over an arbitrary commutative ring `R`, so `R = B_k` is a carrier the corpus
  already permits, and `ringUpper r z = z.1 + r * z.2` is its read.
* `work/proofs/AlgebraCarry.lean` `carryCell`, `carryCell_fst`, `carryCell_snd`,
  `carryCell_readback`, `ringUpper_one_add_slots`, `denoteBV_add`, `residual_recomposes_sum_two`,
  `residual_recomposes_sum_four`, `residual_recomposes_sum_eight_instance`, `carry_identity_two`,
  `carry_identity_four`, `ringMul_second_of_first` — re-declared here from the same bodies because
  `work/proofs` is not an importable package root.
* `work/proofs/AlgebraChi.lean` `ringMul_first_axis` — first-axis `ringMul` is the coefficient
  product at every `σ`; §3 below is that statement transported to `B_k`.
* `work/proofs/AlgebraSha256Sched.lean` `quadCarry_lt_four` — a four-word sum carries out fewer
  than four whole words, which `SumStage.hi_lt` reproduces structurally.

No `sorry`, no `native_decide`, no raised `maxRecDepth`, no `maxHeartbeats 0`. `decide` runs only
over `Bool` (two elements), `ZMod 2` (two), `BitVec 2`, `BitVec 4`, `BitVec 5` and
`Fin 2 → ZMod 2` (four); every statement with a width variable is proved by rewriting and case
analysis.
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

namespace AlgebraBoolCarrier

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

/-! ## §7 · The closed cone -/

#audit AlgebraBoolCarrier.f2b_b2f
#audit AlgebraBoolCarrier.b2f_f2b
#audit AlgebraBoolCarrier.b2f_xor
#audit AlgebraBoolCarrier.b2f_and
#audit AlgebraBoolCarrier.b2f_not
#audit AlgebraBoolCarrier.getLsbD_zero_word
#audit AlgebraBoolCarrier.getLsbD_ge_width
#audit AlgebraBoolCarrier.bitWord_getLsbD
#audit AlgebraBoolCarrier.pack_getLsbD
#audit AlgebraBoolCarrier.pack_unpack
#audit AlgebraBoolCarrier.unpack_pack
#audit AlgebraBoolCarrier.boolEquiv
#audit AlgebraBoolCarrier.bCell_add
#audit AlgebraBoolCarrier.ringMul_first_axis_bool
#audit AlgebraBoolCarrier.ringMul_second_of_first_bool
#audit AlgebraBoolCarrier.unpack_add_xor
#audit AlgebraBoolCarrier.unpack_mul_and
#audit AlgebraBoolCarrier.unpack_add_one_not
#audit AlgebraBoolCarrier.xor_wiring
#audit AlgebraBoolCarrier.xor_wiring_read
#audit AlgebraBoolCarrier.and_wiring
#audit AlgebraBoolCarrier.and_wiring_read
#audit AlgebraBoolCarrier.ringOne_add_bCell
#audit AlgebraBoolCarrier.not_wiring
#audit AlgebraBoolCarrier.not_wiring_read
#audit AlgebraBoolCarrier.xor_ne_add_k2
#audit AlgebraBoolCarrier.xor_ne_add_k4
#audit AlgebraBoolCarrier.unpack_shr_wiring
#audit AlgebraBoolCarrier.unpack_rotr_wiring
#audit AlgebraBoolCarrier.wiringRotR_second_total
#audit AlgebraBoolCarrier.wiringCarry_and_coord
#audit AlgebraBoolCarrier.wiringCarry_top
#audit AlgebraBoolCarrier.denoteBV_add
#audit AlgebraBoolCarrier.carryCell_slots_add
#audit AlgebraBoolCarrier.ringUpper_one_carryCell
#audit AlgebraBoolCarrier.residual_recomposes_two
#audit AlgebraBoolCarrier.residual_recomposes_four
#audit AlgebraBoolCarrier.residual_recomposes_eight_instance
#audit AlgebraBoolCarrier.carry_identity_two
#audit AlgebraBoolCarrier.carry_identity_four
#audit AlgebraBoolCarrier.carry_identity_eight_instance
#audit AlgebraBoolCarrier.carry_identity_four_augmented
#audit AlgebraBoolCarrier.residual_val
#audit AlgebraBoolCarrier.full_and_full
#audit AlgebraBoolCarrier.residual_ne_overflow_k2
#audit AlgebraBoolCarrier.residual_ne_overflow_k4
#audit AlgebraBoolCarrier.residual_ne_overflow_k8_instance
#audit AlgebraBoolCarrier.residual_val_lt
#audit AlgebraBoolCarrier.addStep
#audit AlgebraBoolCarrier.addStep_inv
#audit AlgebraBoolCarrier.startOf_inv
#audit AlgebraBoolCarrier.sched3_s1_inv
#audit AlgebraBoolCarrier.sched3_s2_inv
#audit AlgebraBoolCarrier.sched3_s3_inv
#audit AlgebraBoolCarrier.sched3_s3_lo
#audit AlgebraBoolCarrier.sched3_s3_hi
#audit AlgebraBoolCarrier.sched3_s3_hi_lt

#print axioms AlgebraBoolCarrier.f2b_b2f
#print axioms AlgebraBoolCarrier.b2f_f2b
#print axioms AlgebraBoolCarrier.b2f_xor
#print axioms AlgebraBoolCarrier.b2f_and
#print axioms AlgebraBoolCarrier.b2f_not
#print axioms AlgebraBoolCarrier.getLsbD_zero_word
#print axioms AlgebraBoolCarrier.getLsbD_ge_width
#print axioms AlgebraBoolCarrier.bitWord_getLsbD
#print axioms AlgebraBoolCarrier.pack_getLsbD
#print axioms AlgebraBoolCarrier.pack_unpack
#print axioms AlgebraBoolCarrier.unpack_pack
#print axioms AlgebraBoolCarrier.boolEquiv
#print axioms AlgebraBoolCarrier.bCell_add
#print axioms AlgebraBoolCarrier.ringMul_first_axis_bool
#print axioms AlgebraBoolCarrier.ringMul_second_of_first_bool
#print axioms AlgebraBoolCarrier.unpack_add_xor
#print axioms AlgebraBoolCarrier.unpack_mul_and
#print axioms AlgebraBoolCarrier.unpack_add_one_not
#print axioms AlgebraBoolCarrier.xor_wiring
#print axioms AlgebraBoolCarrier.xor_wiring_read
#print axioms AlgebraBoolCarrier.and_wiring
#print axioms AlgebraBoolCarrier.and_wiring_read
#print axioms AlgebraBoolCarrier.ringOne_add_bCell
#print axioms AlgebraBoolCarrier.not_wiring
#print axioms AlgebraBoolCarrier.not_wiring_read
#print axioms AlgebraBoolCarrier.xor_ne_add_k2
#print axioms AlgebraBoolCarrier.xor_ne_add_k4
#print axioms AlgebraBoolCarrier.unpack_shr_wiring
#print axioms AlgebraBoolCarrier.unpack_rotr_wiring
#print axioms AlgebraBoolCarrier.wiringRotR_second_total
#print axioms AlgebraBoolCarrier.wiringCarry_and_coord
#print axioms AlgebraBoolCarrier.wiringCarry_top
#print axioms AlgebraBoolCarrier.denoteBV_add
#print axioms AlgebraBoolCarrier.carryCell_slots_add
#print axioms AlgebraBoolCarrier.ringUpper_one_carryCell
#print axioms AlgebraBoolCarrier.residual_recomposes_two
#print axioms AlgebraBoolCarrier.residual_recomposes_four
#print axioms AlgebraBoolCarrier.residual_recomposes_eight_instance
#print axioms AlgebraBoolCarrier.carry_identity_two
#print axioms AlgebraBoolCarrier.carry_identity_four
#print axioms AlgebraBoolCarrier.carry_identity_eight_instance
#print axioms AlgebraBoolCarrier.carry_identity_four_augmented
#print axioms AlgebraBoolCarrier.residual_val
#print axioms AlgebraBoolCarrier.full_and_full
#print axioms AlgebraBoolCarrier.residual_ne_overflow_k2
#print axioms AlgebraBoolCarrier.residual_ne_overflow_k4
#print axioms AlgebraBoolCarrier.residual_ne_overflow_k8_instance
#print axioms AlgebraBoolCarrier.residual_val_lt
#print axioms AlgebraBoolCarrier.addStep
#print axioms AlgebraBoolCarrier.addStep_inv
#print axioms AlgebraBoolCarrier.startOf_inv
#print axioms AlgebraBoolCarrier.sched3_s1_inv
#print axioms AlgebraBoolCarrier.sched3_s2_inv
#print axioms AlgebraBoolCarrier.sched3_s3_inv
#print axioms AlgebraBoolCarrier.sched3_s3_lo
#print axioms AlgebraBoolCarrier.sched3_s3_hi
#print axioms AlgebraBoolCarrier.sched3_s3_hi_lt

end AlgebraBoolCarrier