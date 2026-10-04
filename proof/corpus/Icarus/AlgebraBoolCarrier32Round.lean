/-
  AR-60 — the SHA-256 round step on the width-32 Boolean carrier.

  The round step's word functions are defined on `CWord = BoolRing 32 = Fin 32 → ZMod 2`:
  Σ0/Σ1 as the ROTR coordinate wires, Ch/Maj in their F2 normal forms, the word additions through
  the two-input carry step `addStep` (width-32 instances), and the eight-word update. The canonical
  round body of `Apeiron.Crypto.Sha2.compress` is named `roundStep`/`roundStepFun`
  (`Crypto/Sha2.lean:96-120`), `compress_eq_rounds` is the definitional link, and
  `carrierRound_pack`/`carrierRound_coord` are the consistency statements: the packed carrier round
  is the canonical word round at every one of the eight words and every bit.

  Not in scope: the message schedule's injection (`kt`/`wt` are inputs), the sixty-four-round loop,
  the compression feed-forward beyond `compress_eq_rounds`, the digest/store, the executing
  `Cell<i64>` refinement, and any security, collision, preimage, performance or mining claim.
  `lengthBlock`-style length-field behavior is not addressed.

  This file re-declares the needed part of `work/proofs/AlgebraBoolCarrier32.lean` (AR-53, sha256
  ba19b1da739bf400d13bc9676dc5bef3eadca6936127d36a9937fb6d8b0273e0, read-only here) from the same
  bodies, because `work/proofs` is not an importable module root; §1–§6 below are that copy, and
  §7–§11 are the new content. No `sorry`, no `admit`, no axiom, no `native_decide`, no raised
  `maxRecDepth`, no `maxHeartbeats 0`.
-/
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import Mathlib.Data.BitVec
import Mathlib.Tactic
import Lean

set_option autoImplicit false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraBoolCarrier32Round
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
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
/-- The all-ones word of width `k`. -/
def full (k : ℕ) : BitVec k := ~~~((0 : BitVec k))

/-- The all-ones word has representative `2^k - 1`. -/
theorem full_val (k : ℕ) : (full k).toNat = 2 ^ k - 1 := by simp [full]
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
/-- **ROTR wiring at 32**, every amount `n < 32`. -/
theorem unpack_rotr_wiring_32 (n : ℕ) (hn : n < 32) (x : BitVec 32) :
    wiringRotR n (unpack x) = unpack (x.rotateRight n) := unpack_rotr_wiring (k := 32) n hn x

/-- **SHR wiring at 32**, every amount. -/
theorem unpack_shr_wiring_32 (n : ℕ) (x : BitVec 32) :
    wiringShr (unpack x) n = unpack (x >>> n) := unpack_shr_wiring x n

/-! ## §7 · The canonical round step and its two links to `Crypto.Sha2`

`Apeiron.Crypto.Sha2.compress` (`Crypto/Sha2.lean:96-120`) runs its sixty-four rounds as one inline
`List.range 64` fold; the corpus exports no standalone round function. The round body is named here
as `roundStepFun` (function state) and `roundStep` (vector state, the exact fold body), and the two
links are checked:

* `compress_eq_rounds` — `compress h block` is the `roundStep` fold plus the feed-forward
  `h[i] + v[i]`, by kernel `rfl`;
* `roundStep_eq_fun` — the vector `roundStep` is `Vector.ofFn` of the function `roundStepFun` at
  the `getD` view.

The schedule and the loop are not in scope: `kt` and `wt` are inputs. -/

/-- **The canonical round step**, with the round constant and the schedule word explicit
(`Crypto/Sha2.lean:99-119`): `t1 = hh + Σ1(e) + Ch(e,f,g) + kt + wt`, `t2 = Σ0(a) + Maj(a,b,c)`,
and the update `(a,b,c,d,e,f,g,h) ← (t1+t2, a, b, c, d+t1, e, f, g)`. -/
def roundStepFun (s : Fin 8 → Word) (kt wt : Word) : Fin 8 → Word :=
  let a := s 0
  let b := s 1
  let c := s 2
  let d := s 3
  let e := s 4
  let f := s 5
  let g := s 6
  let hh := s 7
  let t1 := hh + Sigma1 e + choose e f g + kt + wt
  let t2 := Sigma0 a + majority a b c
  fun i => match i.val with
    | 0 => t1 + t2
    | 1 => a
    | 2 => b
    | 3 => c
    | 4 => d + t1
    | 5 => e
    | 6 => f
    | _ => g

/-- **The same step on the vector state**, the body of `compress` verbatim. -/
def roundStep (s : State) (kt wt : Word) : State :=
  let a := s.getD 0 0
  let b := s.getD 1 0
  let c := s.getD 2 0
  let d := s.getD 3 0
  let e := s.getD 4 0
  let f := s.getD 5 0
  let g := s.getD 6 0
  let hh := s.getD 7 0
  let t1 := hh + Sigma1 e + choose e f g + kt + wt
  let t2 := Sigma0 a + majority a b c
  Vector.ofFn fun i : Fin 8 =>
    match i.val with
    | 0 => t1 + t2
    | 1 => a
    | 2 => b
    | 3 => c
    | 4 => d + t1
    | 5 => e
    | 6 => f
    | _ => g

/-- **`compress` as a `roundStep` fold.** Definitional: `compress` is this fold, so every later link
composes with the canonical compression's own body. -/
def compressVia (h : State) (block : Block) : State :=
  let w := schedule block
  let v0 : State := h
  let v := (List.range 64).foldl (fun acc t => roundStep acc (K.getD t 0) (w.getD t 0)) v0
  Vector.ofFn fun i : Fin 8 => h.getD i.val 0 + v.getD i.val 0

/-- **`Sha2.compress` is the sixty-four-round `roundStep` fold** with the schedule and constant
tables, plus the feed-forward. Kernel `rfl`: the two bodies are definitionally equal. -/
theorem compress_eq_rounds (h : State) (block : Block) :
    compress h block = compressVia h block := rfl

/-- **The vector round step is the function round step read at `getD` 0.** -/
theorem roundStep_eq_fun (s : State) (kt wt : Word) :
    roundStep s kt wt
      = Vector.ofFn (fun i : Fin 8 => roundStepFun (fun j : Fin 8 => s.getD j.val 0) kt wt i) := by
  rfl

/-- **The canonical `rotr` is `BitVec.rotateRight` at 32 bits**, for every positive amount below
the width. `rotr` reduces the amount mod 32 and fills from `(32 - n) % 32`; `0 < n < 32` makes both
moduli the identity. -/
theorem rotr_eq_rotateRight (x : BitVec 32) {n : ℕ} (h0 : 0 < n) (hn : n < 32) :
    rotr x n = x.rotateRight n := by
  unfold rotr BitVec.rotateRight BitVec.rotateRightAux
  rw [Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt (by omega : 32 - n < 32)]

/-- **Three-term exclusive-or is a coordinate sum on the Boolean carrier.** -/
theorem unpack_zxor3 {k : ℕ} (x y z : BitVec k) :
    unpack (x ^^^ y ^^^ z) = unpack x + unpack y + unpack z := by
  rw [unpack_add_xor x y, unpack_add_xor (x ^^^ y) z]

/-! ## §8 · The round terms on the width-32 Boolean carrier

`CWord = BoolRing 32 = Fin 32 → ZMod 2` with the pointwise ring operations; `CState = Fin 8 → CWord`.
Σ0/Σ1 are the ROTR coordinate wires; Ch/Maj are the F2 normal forms (`Ch = x·y + z + x·z`,
`Maj = x·y + x·z + y·z`); the sums run through the two-input carry step `addStep`, so the word
addition and its integer overflow count stay the objects AR-53 checked. -/

/-- The width-32 Boolean carrier word: `Fin 32 → ZMod 2`, pointwise operations. -/
abbrev CWord : Type := BoolRing32

/-- The eight-word round state on the carrier, indexed `0 … 7` as `a … h`. -/
abbrev CState : Type := Fin 8 → CWord

/-- **Σ0 on the carrier**: the ROTR 2/13/22 coordinate wires, summed in `ZMod 2`. -/
def carrierSigma0 (x : CWord) : CWord :=
  wiringRotR 2 x + wiringRotR 13 x + wiringRotR 22 x

/-- **Σ1 on the carrier**: the ROTR 6/11/25 coordinate wires, summed in `ZMod 2`. -/
def carrierSigma1 (x : CWord) : CWord :=
  wiringRotR 6 x + wiringRotR 11 x + wiringRotR 25 x

/-- **Ch on the carrier, F2 normal form**: `x·y + z + x·z`, the quadratic `(x∧y) ⊕ (¬x∧z)`
read coordinatewise. -/
def carrierCh (x y z : CWord) : CWord := x * y + z + x * z

/-- **Maj on the carrier, F2 normal form**: `x·y + x·z + y·z`. -/
def carrierMaj (x y z : CWord) : CWord := x * y + x * z + y * z

/-- **Σ0's wire composite agrees with the canonical `Sha2.Sigma0` after packing.** -/
theorem carrierSigma0_unpack (x : BitVec 32) :
    carrierSigma0 (unpack x) = unpack (Sigma0 x) := by
  unfold carrierSigma0 Sigma0
  rw [rotr_eq_rotateRight x (n := 2) (by norm_num) (by norm_num),
      rotr_eq_rotateRight x (n := 13) (by norm_num) (by norm_num),
      rotr_eq_rotateRight x (n := 22) (by norm_num) (by norm_num),
      unpack_zxor3]
  rw [← unpack_rotr_wiring_32 2 (by norm_num),
      ← unpack_rotr_wiring_32 13 (by norm_num),
      ← unpack_rotr_wiring_32 22 (by norm_num)]

/-- **Σ1's wire composite agrees with the canonical `Sha2.Sigma1` after packing.** -/
theorem carrierSigma1_unpack (x : BitVec 32) :
    carrierSigma1 (unpack x) = unpack (Sigma1 x) := by
  unfold carrierSigma1 Sigma1
  rw [rotr_eq_rotateRight x (n := 6) (by norm_num) (by norm_num),
      rotr_eq_rotateRight x (n := 11) (by norm_num) (by norm_num),
      rotr_eq_rotateRight x (n := 25) (by norm_num) (by norm_num),
      unpack_zxor3]
  rw [← unpack_rotr_wiring_32 6 (by norm_num),
      ← unpack_rotr_wiring_32 11 (by norm_num),
      ← unpack_rotr_wiring_32 25 (by norm_num)]

/-- **Ch's F2 normal form agrees with the canonical `Sha2.choose` after packing.** The `¬x` needs
the NOT wire `unpack (~~~x) = 1 + unpack x`, then `(1 + x)·z = z + x·z`. -/
theorem carrierCh_unpack (x y z : BitVec 32) :
    carrierCh (unpack x) (unpack y) (unpack z) = unpack (choose x y z) := by
  unfold carrierCh choose
  rw [← unpack_add_xor (x &&& y) (~~~ x &&& z)]
  rw [← unpack_mul_and x y, ← unpack_mul_and (~~~ x) z, ← unpack_add_one_not x]
  ring

/-- **Maj's F2 normal form agrees with the canonical `Sha2.majority` after packing.** -/
theorem carrierMaj_unpack (x y z : BitVec 32) :
    carrierMaj (unpack x) (unpack y) (unpack z) = unpack (majority x y z) := by
  unfold carrierMaj majority
  rw [unpack_zxor3 (x &&& y) (x &&& z) (y &&& z)]
  rw [← unpack_mul_and x y, ← unpack_mul_and x z, ← unpack_mul_and y z]

/-- **Σ0 on the carrier, packed**, is the canonical `Sigma0` of the packed word. -/
theorem pack_carrierSigma0 (x : CWord) :
    pack 32 (carrierSigma0 x) = Sigma0 (pack 32 x) := by
  simpa [unpack_pack_32, pack_unpack_32] using
    congrArg (pack 32) (carrierSigma0_unpack (pack 32 x))

/-- **Σ1 on the carrier, packed**, is the canonical `Sigma1` of the packed word. -/
theorem pack_carrierSigma1 (x : CWord) :
    pack 32 (carrierSigma1 x) = Sigma1 (pack 32 x) := by
  simpa [unpack_pack_32, pack_unpack_32] using
    congrArg (pack 32) (carrierSigma1_unpack (pack 32 x))

/-- **Ch on the carrier, packed**, is the canonical `choose` of the packed words. -/
theorem pack_carrierCh (x y z : CWord) :
    pack 32 (carrierCh x y z) = choose (pack 32 x) (pack 32 y) (pack 32 z) := by
  simpa [unpack_pack_32, pack_unpack_32] using
    congrArg (pack 32) (carrierCh_unpack (pack 32 x) (pack 32 y) (pack 32 z))

/-- **Maj on the carrier, packed**, is the canonical `majority` of the packed words. -/
theorem pack_carrierMaj (x y z : CWord) :
    pack 32 (carrierMaj x y z) = majority (pack 32 x) (pack 32 y) (pack 32 z) := by
  simpa [unpack_pack_32, pack_unpack_32] using
    congrArg (pack 32) (carrierMaj_unpack (pack 32 x) (pack 32 y) (pack 32 z))

/-- **Carrier-word addition through the two-input carry step**: pack both words, run one `addStep`,
read the represented word. The stage's integer overflow count is discarded here; `carrierT1Stage`
keeps the full chain when the count is wanted. -/
def carrierAdd (x y : CWord) : CWord :=
  unpack (addStep (startOf (pack 32 x)) (pack 32 y)).lo

/-- **The carry-step addition, packed**, is the canonical 32-bit wrapping sum. -/
theorem pack_carrierAdd (x y : CWord) :
    pack 32 (carrierAdd x y) = pack 32 x + pack 32 y := by
  simp [carrierAdd, addStep, startOf, pack_unpack_32]

/-- **T1's five-term chain** `hh + Σ1(e) + Ch(e,f,g) + K[t] + W[t]` as four two-input carry
stages, with the integer overflow count kept in the `SumStage`. -/
def carrierT1Stage (hh se ch kt wt : CWord) : SumStage 32 5 :=
  addStep (addStep (addStep (addStep (startOf (pack 32 hh)) (pack 32 se)) (pack 32 ch))
    (pack 32 kt)) (pack 32 wt)

/-- **T1 on the carrier**: the Σ1 wire and the Ch normal form at `e f g`, then the carry chain. -/
def carrierT1 (hh e f g kt wt : CWord) : CWord :=
  unpack (carrierT1Stage hh (carrierSigma1 e) (carrierCh e f g) kt wt).lo

/-- **The represented word of the five-term chain is the wrapping five-word sum.** -/
theorem carrierT1Stage_lo (hh se ch kt wt : CWord) :
    (carrierT1Stage hh se ch kt wt).lo
      = pack 32 hh + pack 32 se + pack 32 ch + pack 32 kt + pack 32 wt := rfl

/-- **T1 on the carrier, packed**, is the canonical five-term word expression. -/
theorem pack_carrierT1 (hh e f g kt wt : CWord) :
    pack 32 (carrierT1 hh e f g kt wt)
      = pack 32 hh + Sigma1 (pack 32 e) + choose (pack 32 e) (pack 32 f) (pack 32 g)
        + pack 32 kt + pack 32 wt := by
  rw [carrierT1, carrierT1Stage_lo, pack_unpack_32, pack_carrierSigma1, pack_carrierCh]

/-- **Every carry stage preserves the schedule invariant**, so the five-term chain's integer
overflow count represents the running total exactly (`SchedInv`), the AR-53 invariant at `n = 5`. -/
theorem carrierT1Stage_inv (hh se ch kt wt : CWord) :
    SchedInv 32 5
      ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat
        + (pack 32 kt).toNat + (pack 32 wt).toNat) (carrierT1Stage hh se ch kt wt) := by
  unfold carrierT1Stage
  exact addStep_inv (addStep_inv (addStep_inv (addStep_inv (startOf_inv (pack 32 hh)))))

/-- **The five-term chain's integer overflow count** is the whole-word quotient of the five-term
integer total, the component with no slot in a 32-bit word. -/
theorem carrierT1Stage_hi (hh se ch kt wt : CWord) :
    (carrierT1Stage hh se ch kt wt).hi
      = ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat
          + (pack 32 kt).toNat + (pack 32 wt).toNat) / 2 ^ 32 := by
  have hinv := carrierT1Stage_inv hh se ch kt wt
  simp only [SchedInv] at hinv
  have hz : (0 : ℕ) < 2 ^ 32 := two_pow_pos 32
  have hlt : (carrierT1Stage hh se ch kt wt).lo.toNat < 2 ^ 32 :=
    (carrierT1Stage hh se ch kt wt).lo.isLt
  have hmod : ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat
        + (pack 32 kt).toNat + (pack 32 wt).toNat) % 2 ^ 32
      = (carrierT1Stage hh se ch kt wt).lo.toNat := by
    have e := Nat.add_mul_mod_self_left (carrierT1Stage hh se ch kt wt).lo.toNat (2 ^ 32)
      (carrierT1Stage hh se ch kt wt).hi
    rw [← hinv, Nat.mod_eq_of_lt hlt] at e
    exact e
  have key := Nat.mod_add_div ((pack 32 hh).toNat + (pack 32 se).toNat + (pack 32 ch).toNat
        + (pack 32 kt).toNat + (pack 32 wt).toNat) (2 ^ 32)
  rw [hmod] at key
  refine Nat.mul_left_cancel hz ?_
  omega

/-- **T2 on the carrier**: `Σ0(a) + Maj(a,b,c)` through the carry step. -/
def carrierT2 (a b c : CWord) : CWord :=
  carrierAdd (carrierSigma0 a) (carrierMaj a b c)

/-- **T2 on the carrier, packed**, is the canonical `Σ0(a) + Maj(a,b,c)`. -/
theorem pack_carrierT2 (a b c : CWord) :
    pack 32 (carrierT2 a b c)
      = Sigma0 (pack 32 a) + majority (pack 32 a) (pack 32 b) (pack 32 c) := by
  rw [carrierT2, pack_carrierAdd, pack_carrierSigma0, pack_carrierMaj]

/-- **The carrier round step**: `t1 = Σ1(e) + Ch(e,f,g) + hh + K + W` through the carry chain,
`t2 = Σ0(a) + Maj(a,b,c)`, and the eight-word update
`(a,b,c,d,e,f,g,h) ← (t1+t2, a, b, c, d+t1, e, f, g)`. Every word operation is total. -/
def carrierRound (s : CState) (kt wt : CWord) : CState :=
  let a := s 0
  let b := s 1
  let c := s 2
  let d := s 3
  let e := s 4
  let f := s 5
  let g := s 6
  let hh := s 7
  let t1 := carrierT1 hh e f g kt wt
  let t2 := carrierT2 a b c
  fun i => match i.val with
    | 0 => carrierAdd t1 t2
    | 1 => a
    | 2 => b
    | 3 => c
    | 4 => carrierAdd d t1
    | 5 => e
    | 6 => f
    | _ => g

/-- **The consistency statement, per element.** For every client state, constant and schedule word
on the carrier, packing the carrier round's word `i` gives the canonical round step's word `i` on
the packed inputs: the carrier dataflow (wires, F2 normal forms, carry-step sums) computes the
canonical `Crypto.Sha2` round at each of the eight coordinates. -/
theorem carrierRound_pack (s : CState) (kt wt : CWord) (i : Fin 8) :
    pack 32 (carrierRound s kt wt i)
      = roundStepFun (fun j : Fin 8 => pack 32 (s j)) (pack 32 kt) (pack 32 wt) i := by
  fin_cases i <;>
    simp only [carrierRound, roundStepFun, pack_carrierAdd, pack_carrierT1, pack_carrierT2]

/-- **The consistency statement, per element and per bit.** Bit `j` of the packed carrier round's
word `i` is the `j`-th bit of the canonical word round's word `i`. -/
theorem carrierRound_coord (s : CState) (kt wt : CWord) (i : Fin 8) (j : Fin 32) :
    carrierRound s kt wt i j
      = b2f ((roundStepFun (fun l : Fin 8 => pack 32 (s l)) (pack 32 kt) (pack 32 wt) i).getLsbD
          j.val) := by
  have h := congrArg (fun w : BitVec 32 => w.getLsbD j.val) (carrierRound_pack s kt wt i)
  rw [pack_getLsbD_32] at h
  rw [← h, b2f_f2b]

/-! ## §9 · Face verdict: no step of this composition reads σ -/

/-- **Every multiplicative step the round uses is a first-axis `ringMul`, at every face.** The two
products of `carrierCh` and the three of `carrierMaj` are `(ringMul σ (bCell ·) (bCell ·)).1` for
every `σ : CWord`; the additions, the ROTR wires and the `BitVec` carry step take no face. This is
AR-37's any-face wiring statement at the composition: the round pins no σ. -/
theorem round_products_face_free (σ : CWord) (x y z : CWord) :
    (ringMul σ (bCell x) (bCell y)).1 = x * y ∧
    (ringMul σ (bCell x) (bCell z)).1 = x * z ∧
    (ringMul σ (bCell y) (bCell z)).1 = y * z := by
  refine ⟨?_, ?_, ?_⟩ <;> rw [ringMul_first_axis_bool, bCell_fst]

/-! ## §10 · Reachable instances -/

/-- The zero vector un-packs to the zero word. -/
theorem unpack_zero_32 : unpack (0 : BitVec 32) = 0 := by
  funext i
  simp [unpack, b2f, getLsbD_zero_word]

/-- The zero word packs to the zero vector. -/
theorem pack_zero_32 : pack 32 (0 : CWord) = 0 := by
  rw [← unpack_zero_32, pack_unpack_32]

/-- A rotation wire sends the zero vector to the zero vector. -/
theorem wiringRotR_zero {k : ℕ} (n : ℕ) : wiringRotR n (0 : BoolRing k) = 0 := by
  funext i
  simp [wiringRotR]

/-- **A reachable round state**: on the all-zero carrier state, zero constant and zero schedule
word, every one of the eight carrier words is zero, coordinatewise. -/
theorem carrierRound_zero (i : Fin 8) (j : Fin 32) :
    carrierRound (fun _ : Fin 8 => 0) 0 0 i j = 0 := by
  fin_cases i <;>
    simp [carrierRound, carrierT1, carrierT2, carrierT1Stage, carrierAdd, carrierSigma0,
      carrierSigma1, carrierCh, carrierMaj, wiringRotR_zero, addStep, startOf, pack_zero_32,
      pack_unpack_32, unpack_zero_32, unpack, b2f, getLsbD_zero_word]

/-- **A reachable nonzero overflow count**: the five-term carry chain at five all-ones carrier
words represents the integer total `5·(2^32 − 1)` and carries out `4` whole 32-bit units, so the
count component is exercised on a nonzero state. -/
theorem carrierT1Stage_full_hi :
    (carrierT1Stage (unpack (full 32)) (unpack (full 32)) (unpack (full 32))
      (unpack (full 32)) (unpack (full 32))).hi = 4 := by
  rw [carrierT1Stage_hi]
  simp only [pack_unpack_32, full_val]
  norm_num

/-- The represented word of the same instance is the wrapping five-word all-ones sum. -/
theorem carrierT1Stage_full_lo :
    (carrierT1Stage (unpack (full 32)) (unpack (full 32)) (unpack (full 32))
      (unpack (full 32)) (unpack (full 32))).lo
      = full 32 + full 32 + full 32 + full 32 + full 32 := by
  rw [carrierT1Stage_lo]
  simp only [pack_unpack_32]

/-! ## §11 · The closed cone -/

#audit AlgebraBoolCarrier32Round.BoolCell
#audit AlgebraBoolCarrier32Round.BoolCell32
#audit AlgebraBoolCarrier32Round.BoolRing
#audit AlgebraBoolCarrier32Round.BoolRing32
#audit AlgebraBoolCarrier32Round.CState
#audit AlgebraBoolCarrier32Round.CWord
#audit AlgebraBoolCarrier32Round.SchedInv
#audit AlgebraBoolCarrier32Round.SumStage
#audit AlgebraBoolCarrier32Round.addStep
#audit AlgebraBoolCarrier32Round.addStep_inv
#audit AlgebraBoolCarrier32Round.b2f
#audit AlgebraBoolCarrier32Round.b2f_and
#audit AlgebraBoolCarrier32Round.b2f_f2b
#audit AlgebraBoolCarrier32Round.b2f_not
#audit AlgebraBoolCarrier32Round.b2f_xor
#audit AlgebraBoolCarrier32Round.bCell
#audit AlgebraBoolCarrier32Round.bCell_add
#audit AlgebraBoolCarrier32Round.bCell_fst
#audit AlgebraBoolCarrier32Round.bCell_snd
#audit AlgebraBoolCarrier32Round.bitWord
#audit AlgebraBoolCarrier32Round.bitWord_getLsbD
#audit AlgebraBoolCarrier32Round.boolEquiv
#audit AlgebraBoolCarrier32Round.carrierAdd
#audit AlgebraBoolCarrier32Round.carrierCh
#audit AlgebraBoolCarrier32Round.carrierCh_unpack
#audit AlgebraBoolCarrier32Round.carrierMaj
#audit AlgebraBoolCarrier32Round.carrierMaj_unpack
#audit AlgebraBoolCarrier32Round.carrierRound
#audit AlgebraBoolCarrier32Round.carrierRound_coord
#audit AlgebraBoolCarrier32Round.carrierRound_pack
#audit AlgebraBoolCarrier32Round.carrierRound_zero
#audit AlgebraBoolCarrier32Round.carrierSigma0
#audit AlgebraBoolCarrier32Round.carrierSigma0_unpack
#audit AlgebraBoolCarrier32Round.carrierSigma1
#audit AlgebraBoolCarrier32Round.carrierSigma1_unpack
#audit AlgebraBoolCarrier32Round.carrierT1
#audit AlgebraBoolCarrier32Round.carrierT1Stage
#audit AlgebraBoolCarrier32Round.carrierT1Stage_full_hi
#audit AlgebraBoolCarrier32Round.carrierT1Stage_full_lo
#audit AlgebraBoolCarrier32Round.carrierT1Stage_hi
#audit AlgebraBoolCarrier32Round.carrierT1Stage_inv
#audit AlgebraBoolCarrier32Round.carrierT1Stage_lo
#audit AlgebraBoolCarrier32Round.carrierT2
#audit AlgebraBoolCarrier32Round.compressVia
#audit AlgebraBoolCarrier32Round.compress_eq_rounds
#audit AlgebraBoolCarrier32Round.f2b
#audit AlgebraBoolCarrier32Round.f2b_b2f
#audit AlgebraBoolCarrier32Round.full
#audit AlgebraBoolCarrier32Round.full_val
#audit AlgebraBoolCarrier32Round.getLsbD_ge_width
#audit AlgebraBoolCarrier32Round.getLsbD_not_eq
#audit AlgebraBoolCarrier32Round.getLsbD_zero_word
#audit AlgebraBoolCarrier32Round.one_lt_two_pow
#audit AlgebraBoolCarrier32Round.pack
#audit AlgebraBoolCarrier32Round.pack_carrierAdd
#audit AlgebraBoolCarrier32Round.pack_carrierCh
#audit AlgebraBoolCarrier32Round.pack_carrierMaj
#audit AlgebraBoolCarrier32Round.pack_carrierSigma0
#audit AlgebraBoolCarrier32Round.pack_carrierSigma1
#audit AlgebraBoolCarrier32Round.pack_carrierT1
#audit AlgebraBoolCarrier32Round.pack_carrierT2
#audit AlgebraBoolCarrier32Round.pack_getLsbD
#audit AlgebraBoolCarrier32Round.pack_getLsbD_32
#audit AlgebraBoolCarrier32Round.pack_unpack
#audit AlgebraBoolCarrier32Round.pack_unpack_32
#audit AlgebraBoolCarrier32Round.pack_zero_32
#audit AlgebraBoolCarrier32Round.ringMul_first_axis_bool
#audit AlgebraBoolCarrier32Round.ringMul_second_of_first_bool
#audit AlgebraBoolCarrier32Round.rotr_eq_rotateRight
#audit AlgebraBoolCarrier32Round.roundStep
#audit AlgebraBoolCarrier32Round.roundStepFun
#audit AlgebraBoolCarrier32Round.roundStep_eq_fun
#audit AlgebraBoolCarrier32Round.round_products_face_free
#audit AlgebraBoolCarrier32Round.startOf
#audit AlgebraBoolCarrier32Round.startOf_inv
#audit AlgebraBoolCarrier32Round.two_pow_pos
#audit AlgebraBoolCarrier32Round.unpack
#audit AlgebraBoolCarrier32Round.unpack_add_one_not
#audit AlgebraBoolCarrier32Round.unpack_add_xor
#audit AlgebraBoolCarrier32Round.unpack_mul_and
#audit AlgebraBoolCarrier32Round.unpack_pack
#audit AlgebraBoolCarrier32Round.unpack_pack_32
#audit AlgebraBoolCarrier32Round.unpack_rotr_wiring
#audit AlgebraBoolCarrier32Round.unpack_rotr_wiring_32
#audit AlgebraBoolCarrier32Round.unpack_shr_wiring
#audit AlgebraBoolCarrier32Round.unpack_shr_wiring_32
#audit AlgebraBoolCarrier32Round.unpack_zero_32
#audit AlgebraBoolCarrier32Round.unpack_zxor3
#audit AlgebraBoolCarrier32Round.wiringRotR
#audit AlgebraBoolCarrier32Round.wiringRotR_second_total
#audit AlgebraBoolCarrier32Round.wiringRotR_zero
#audit AlgebraBoolCarrier32Round.wiringShr

#print axioms AlgebraBoolCarrier32Round.BoolCell
#print axioms AlgebraBoolCarrier32Round.BoolCell32
#print axioms AlgebraBoolCarrier32Round.BoolRing
#print axioms AlgebraBoolCarrier32Round.BoolRing32
#print axioms AlgebraBoolCarrier32Round.CState
#print axioms AlgebraBoolCarrier32Round.CWord
#print axioms AlgebraBoolCarrier32Round.SchedInv
#print axioms AlgebraBoolCarrier32Round.SumStage
#print axioms AlgebraBoolCarrier32Round.addStep
#print axioms AlgebraBoolCarrier32Round.addStep_inv
#print axioms AlgebraBoolCarrier32Round.b2f
#print axioms AlgebraBoolCarrier32Round.b2f_and
#print axioms AlgebraBoolCarrier32Round.b2f_f2b
#print axioms AlgebraBoolCarrier32Round.b2f_not
#print axioms AlgebraBoolCarrier32Round.b2f_xor
#print axioms AlgebraBoolCarrier32Round.bCell
#print axioms AlgebraBoolCarrier32Round.bCell_add
#print axioms AlgebraBoolCarrier32Round.bCell_fst
#print axioms AlgebraBoolCarrier32Round.bCell_snd
#print axioms AlgebraBoolCarrier32Round.bitWord
#print axioms AlgebraBoolCarrier32Round.bitWord_getLsbD
#print axioms AlgebraBoolCarrier32Round.boolEquiv
#print axioms AlgebraBoolCarrier32Round.carrierAdd
#print axioms AlgebraBoolCarrier32Round.carrierCh
#print axioms AlgebraBoolCarrier32Round.carrierCh_unpack
#print axioms AlgebraBoolCarrier32Round.carrierMaj
#print axioms AlgebraBoolCarrier32Round.carrierMaj_unpack
#print axioms AlgebraBoolCarrier32Round.carrierRound
#print axioms AlgebraBoolCarrier32Round.carrierRound_coord
#print axioms AlgebraBoolCarrier32Round.carrierRound_pack
#print axioms AlgebraBoolCarrier32Round.carrierRound_zero
#print axioms AlgebraBoolCarrier32Round.carrierSigma0
#print axioms AlgebraBoolCarrier32Round.carrierSigma0_unpack
#print axioms AlgebraBoolCarrier32Round.carrierSigma1
#print axioms AlgebraBoolCarrier32Round.carrierSigma1_unpack
#print axioms AlgebraBoolCarrier32Round.carrierT1
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage_full_hi
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage_full_lo
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage_hi
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage_inv
#print axioms AlgebraBoolCarrier32Round.carrierT1Stage_lo
#print axioms AlgebraBoolCarrier32Round.carrierT2
#print axioms AlgebraBoolCarrier32Round.compressVia
#print axioms AlgebraBoolCarrier32Round.compress_eq_rounds
#print axioms AlgebraBoolCarrier32Round.f2b
#print axioms AlgebraBoolCarrier32Round.f2b_b2f
#print axioms AlgebraBoolCarrier32Round.full
#print axioms AlgebraBoolCarrier32Round.full_val
#print axioms AlgebraBoolCarrier32Round.getLsbD_ge_width
#print axioms AlgebraBoolCarrier32Round.getLsbD_not_eq
#print axioms AlgebraBoolCarrier32Round.getLsbD_zero_word
#print axioms AlgebraBoolCarrier32Round.one_lt_two_pow
#print axioms AlgebraBoolCarrier32Round.pack
#print axioms AlgebraBoolCarrier32Round.pack_carrierAdd
#print axioms AlgebraBoolCarrier32Round.pack_carrierCh
#print axioms AlgebraBoolCarrier32Round.pack_carrierMaj
#print axioms AlgebraBoolCarrier32Round.pack_carrierSigma0
#print axioms AlgebraBoolCarrier32Round.pack_carrierSigma1
#print axioms AlgebraBoolCarrier32Round.pack_carrierT1
#print axioms AlgebraBoolCarrier32Round.pack_carrierT2
#print axioms AlgebraBoolCarrier32Round.pack_getLsbD
#print axioms AlgebraBoolCarrier32Round.pack_getLsbD_32
#print axioms AlgebraBoolCarrier32Round.pack_unpack
#print axioms AlgebraBoolCarrier32Round.pack_unpack_32
#print axioms AlgebraBoolCarrier32Round.pack_zero_32
#print axioms AlgebraBoolCarrier32Round.ringMul_first_axis_bool
#print axioms AlgebraBoolCarrier32Round.ringMul_second_of_first_bool
#print axioms AlgebraBoolCarrier32Round.rotr_eq_rotateRight
#print axioms AlgebraBoolCarrier32Round.roundStep
#print axioms AlgebraBoolCarrier32Round.roundStepFun
#print axioms AlgebraBoolCarrier32Round.roundStep_eq_fun
#print axioms AlgebraBoolCarrier32Round.round_products_face_free
#print axioms AlgebraBoolCarrier32Round.startOf
#print axioms AlgebraBoolCarrier32Round.startOf_inv
#print axioms AlgebraBoolCarrier32Round.two_pow_pos
#print axioms AlgebraBoolCarrier32Round.unpack
#print axioms AlgebraBoolCarrier32Round.unpack_add_one_not
#print axioms AlgebraBoolCarrier32Round.unpack_add_xor
#print axioms AlgebraBoolCarrier32Round.unpack_mul_and
#print axioms AlgebraBoolCarrier32Round.unpack_pack
#print axioms AlgebraBoolCarrier32Round.unpack_pack_32
#print axioms AlgebraBoolCarrier32Round.unpack_rotr_wiring
#print axioms AlgebraBoolCarrier32Round.unpack_rotr_wiring_32
#print axioms AlgebraBoolCarrier32Round.unpack_shr_wiring
#print axioms AlgebraBoolCarrier32Round.unpack_shr_wiring_32
#print axioms AlgebraBoolCarrier32Round.unpack_zero_32
#print axioms AlgebraBoolCarrier32Round.unpack_zxor3
#print axioms AlgebraBoolCarrier32Round.wiringRotR
#print axioms AlgebraBoolCarrier32Round.wiringRotR_second_total
#print axioms AlgebraBoolCarrier32Round.wiringRotR_zero
#print axioms AlgebraBoolCarrier32Round.wiringShr

end AlgebraBoolCarrier32Round
