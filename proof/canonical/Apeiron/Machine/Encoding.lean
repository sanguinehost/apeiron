import Apeiron.Linter.Notation
import Mathlib.Data.Int.DivMod
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.List.Basic

/-!
# What an encoder must realise

@name Machine Encoding
@grade proved

An abstract instruction is a total function on machine state. A target encoder is a pure map
from that instruction to bytes, and the bytes are correct when the hardware's own partial or
divergent operation, wrapped in the encoder's fixed guard sequence, computes the abstract
op's total semantics. This file proves the guard equations that derivation leans on, at the
level where they are arithmetic. What the hardware does with the guarded operands is read
from vendor manuals and stays an assumption; the file says which value the guard forces,
not what silicon is.

Three results:

- **Total division.** The abstract `divT`/`tmodT` are total, with `divT x 0 = x` and
  `tmodT x 0 = 0` -- the convention `../song-ref`'s `emitDivideModuloTail` states and its
  prover's `divT`/`modT` share. The measured x86 guard forces a zero divisor to one before
  the machine divide. `guard_realises_divT` and `guard_realises_tmodT` say the guarded
  machine divide equals the total op, so the encoding realises the total semantics and the
  fault input has no representing state.

- **The chunked fold.** A reduce over a span, split into contiguous chunks folded
  independently from the merge's identity and then combined in order, equals the sequential
  fold: `foldl_chunks`, from associativity and identity alone. This licenses a worker pool
  or a SIMD lane split as an *encoding schedule* of the one abstract fold -- the instruction
  set does not grow to name the schedule. Reordering the combine itself needs commutativity
  as well, and that half is `Table.build_gauge_invariance`, already in the corpus; this file
  adds only the contiguous-chunk step it does not state.

- **The packed predicate readout.** A lane-parallel equality probe ends by packing one
  boolean per lane into the bits of one machine word. `testBit_pack` says the packing is
  lossless: bit `i` of the packed word is exactly lane `i`'s boolean. So a 16-lane SIMD
  probe with a movemask readout is an encoding of sixteen scalar equality predicates, and
  nothing else.

## What stays an assumption

That x86 `idiv` on a nonzero, non-overflow pair computes truncation toward zero, that
`pmovmskb` packs the lane sign bits in lane order, and every other reading of a vendor
manual. Those are claims about silicon, checkable against the manuals and differential
tests, not provable here.
-/

namespace Apeiron
namespace Machine

/-- Total truncating division: the machine's division made total by convention.

    (x d : ℤ)                            [dividend and divisor, integers]
    if d = 0 then x else Int.tdiv x d    [the quotient: truncation toward zero, and a zero
                                          divisor returns the dividend unchanged]

    `Int.tdiv` truncates toward zero, the hardware convention. The collapse: the fault input
    has no representing state, because the op is a function there like everywhere else. -/
def divT (x d : ℤ) : ℤ := if d = 0 then x else Int.tdiv x d

/-- Total truncating remainder, `divT`'s partner.

    (x d : ℤ)                            [dividend and divisor, integers]
    if d = 0 then 0 else Int.tmod x d    [the remainder: a zero divisor returns zero]

    So: `divT x 0 = x` with remainder `0`, one convention for both reads of the divide. -/
def tmodT (x d : ℤ) : ℤ := if d = 0 then 0 else Int.tmod x d

/-- The emitted guard realises total division.

    (x d : ℤ)                                        [dividend and divisor]
    Int.tdiv x (if d = 0 then 1 else d) = divT x d   [the machine divide, its divisor guarded
                                                      from zero to one, is the total op]

    So: `emitDivideModuloTail`'s `cmp/sbb/and/add` sequence forces a zero divisor to one, and
    dividing by one returns the dividend -- the encoding computes `divT`, and divide-by-zero
    is not a fault path but an input the abstract op already maps. -/
theorem guard_realises_divT (x d : ℤ) :
    Int.tdiv x (if d = 0 then 1 else d) = divT x d := by
  by_cases h : d = 0 <;> simp [divT, h]

/-- The same guard realises the total remainder.

    (x d : ℤ)                                          [dividend and divisor]
    Int.tmod x (if d = 0 then 1 else d) = tmodT x d    [the machine remainder, its divisor
                                                        guarded from zero to one, is the
                                                        total op]

    So: the one guard serves both reads of the machine divide. -/
theorem guard_realises_tmodT (x d : ℤ) :
    Int.tmod x (if d = 0 then 1 else d) = tmodT x d := by
  by_cases h : d = 0 <;> simp [tmodT, h]

section Fold

variable {V : Type} (f : V → V → V) (e : V)

/-- Folding a chunk from a running value merges in the chunk's own fold.

    (f : V → V → V) (e : V)                              [a merge and its identity]
    (hassoc : ∀ a b c, f (f a b) c = f a (f b c))        [the merge is associative]
    (hid : ∀ a, f e a = a)                               [`e` is a left identity]
    (hidr : ∀ a, f a e = a)                              [and a right identity]
    (acc : V) (c : List V)                               [a running value and one chunk]

    `List` [a finite sequence of values, in order]
    `foldl` [run the merge left to right over the sequence from a start value]
    c.foldl f acc = f acc (c.foldl f e)                  [folding the chunk from the running
                                                          value equals merging in the chunk's
                                                          independent fold]

    So: a chunk folded independently from the identity can be merged in later -- the algebraic
    step every partial-per-worker scheme rests on. -/
theorem foldl_from_acc
    (hassoc : ∀ a b c, f (f a b) c = f a (f b c)) (hid : ∀ a, f e a = a)
    (hidr : ∀ a, f a e = a) (acc : V) (c : List V) :
    c.foldl f acc = f acc (c.foldl f e) := by
  induction c generalizing acc with
  | nil => simp [List.foldl, hidr]
  | cons v rest ih =>
    simp only [List.foldl]
    rw [ih (f acc v), ih (f e v), hid, hassoc]

/-- A chunked fold is the sequential fold.

    (f : V → V → V) (e : V)                              [a merge and its identity]
    (hassoc : ∀ a b c, f (f a b) c = f a (f b c))        [the merge is associative]
    (hid : ∀ a, f e a = a)                               [`e` is a left identity]
    (hidr : ∀ a, f a e = a)                              [and a right identity]
    (chunks : List (List V))                             [the span, split into contiguous
                                                          chunks]

    `List` [a finite sequence of values, in order]
    `foldl` [run the merge left to right over the sequence from a start value]
    `map` [apply a function to every element]
    `flatten` [the inner lists joined into one, in order]
    (chunks.map (fun c => c.foldl f e)).foldl f e
      = chunks.flatten.foldl f e                         [fold each chunk from the identity,
                                                          then fold the partials in order: the
                                                          same value as one sequential fold
                                                          over the flattened span]

    So: a worker pool over disjoint contiguous chunks with an in-order combine of the
    partials, and equally a SIMD lane split, is a schedule of the one abstract fold, not a
    second operation. Associativity and identity are the whole cost here; reordering the
    combine itself additionally needs commutativity, which is `Table.build_gauge_invariance`'s
    hypothesis pair. -/
theorem foldl_chunks
    (hassoc : ∀ a b c, f (f a b) c = f a (f b c)) (hid : ∀ a, f e a = a)
    (hidr : ∀ a, f a e = a) (chunks : List (List V)) :
    (chunks.map (fun c => c.foldl f e)).foldl f e = chunks.flatten.foldl f e := by
  induction chunks with
  | nil => rfl
  | cons c rest ih =>
    simp only [List.map_cons, List.flatten_cons, List.foldl_cons, List.foldl_append, hid]
    rw [foldl_from_acc f e hassoc hid hidr (c.foldl f e) (rest.map (fun c => c.foldl f e)), ih,
        ← foldl_from_acc f e hassoc hid hidr (c.foldl f e) rest.flatten]

end Fold

/-- Pack one boolean per lane into the bits of one number, lane 0 lowest.

    List Bool → ℕ                        [takes the lane booleans, gives the packed word]

    `List` [a finite sequence of values, in order]
    `Bool` [a boolean, one lane's predicate]
    `ℕ` [a natural number, the packed machine word]

    Lane order is the bit order: the head is bit 0, the rest shift up by one place. -/
def pack : List Bool → ℕ
  | [] => 0
  | b :: r => (if b then 1 else 0) + 2 * pack r

/-- The packed readout is lossless: bit i is lane i.

    (l : List Bool) (i : ℕ)                        [the lane booleans and a lane index]
    (pack l).testBit i = l.getD i false            [reading bit i of the packed word is
                                                    reading lane i, absent lanes false]

    `List` [a finite sequence of values, in order]
    `Bool` [a boolean, one lane's predicate]
    `pack` [the lane booleans packed into the bits of one number, lane 0 lowest]
    `testBit` [read one bit of a number]
    `getD` [read one position of a sequence, with a default beyond its end]

    So: a movemask readout of a lane-parallel equality probe carries exactly the per-lane
    predicates -- the SIMD form is an encoding of the scalar predicates. -/
theorem testBit_pack (l : List Bool) (i : ℕ) :
    (pack l).testBit i = l.getD i false := by
  induction l generalizing i with
  | nil => simp [pack]
  | cons b r ih =>
    cases i with
    | zero =>
      cases b <;> simp [pack, Nat.testBit_zero, Nat.add_mul_mod_self_left]
    | succ i =>
      have hdiv : ((if b then 1 else 0) + 2 * pack r) / 2 = pack r := by
        cases b <;> (simp; try omega)
      calc (pack (b :: r)).testBit (i + 1)
          = ((pack (b :: r)) / 2).testBit i := Nat.testBit_add_one _ _
        _ = (pack r).testBit i := by
            rw [show pack (b :: r) = (if b then 1 else 0) + 2 * pack r from rfl, hdiv]
        _ = r.getD i false := ih i
        _ = (b :: r).getD (i + 1) false := by simp

end Machine
end Apeiron
