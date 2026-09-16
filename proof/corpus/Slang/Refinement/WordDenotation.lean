import Slang.Refinement.WordRing
import Apeiron.Trichotomy.AnyRing

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Refinement

open Slang

private instance wordCommRing : CommRing Word where
  zero := wordRing.zero
  add := wordRing.add
  neg := wordRing.neg
  sub := wordRing.sub
  mul := wordRing.mul
  one := wordRing.one
  add_assoc := wordRing.add_assoc
  zero_add := wordRing.add_zero_left
  add_zero := wordRing.add_zero_right
  add_comm := wordRing.add_comm
  -- Temporal O(count) `Word.add` calls: `Nat.rec` applies one addition per decrement of `count`.
  -- Spatial O(count) recursion depth for the `Nat.rec`.
  nsmul := fun count value => Nat.rec wordRing.zero (fun _ total => wordRing.add total value) count
  nsmul_zero := fun _ => rfl
  nsmul_succ := fun _ _ => rfl
  neg_add_cancel := fun value => Eq.trans (wordRing.add_comm _ _) (wordRing.add_neg_right value)
  sub_eq_add_neg := wordRing.sub_eq_add_neg
  -- Temporal O(count) `Word.add` calls: `.ofNat count` adds `count` times and `.negSucc count` adds
  -- `count + 1` times before negating. Spatial O(count) recursion depth for each `Nat.rec`.
  zsmul := fun integer value =>
    match integer with
    | .ofNat count => Nat.rec wordRing.zero (fun _ total => wordRing.add total value) count
    | .negSucc count => wordRing.neg
        (Nat.rec wordRing.zero (fun _ total => wordRing.add total value) (count + 1))
  zsmul_zero' := fun _ => rfl
  zsmul_succ' := fun _ _ => rfl
  zsmul_neg' := fun _ _ => rfl
  mul_assoc := wordRing.mul_assoc
  one_mul := wordRing.mul_one_left
  mul_one := wordRing.mul_one_right
  zero_mul := by
    intro value
    change Word.mul (Word.ofNat 0) value = Word.ofNat 0
    apply Fin.ext
    exact congrArg (fun natural => natural % wordModulus) (Nat.zero_mul value.val)
  mul_zero := by
    intro value
    change Word.mul value (Word.ofNat 0) = Word.ofNat 0
    apply Fin.ext
    exact congrArg (fun natural => natural % wordModulus) (Nat.mul_zero value.val)
  left_distrib := wordRing.mul_add
  right_distrib := wordRing.add_mul
  mul_comm := wordRing.mul_comm

/-- The bounded representative re-embedded through the constructive word ring. -/
def wordDenote (word : Word) : Word := wordRing.ofNat word.val

theorem wordDenote_eq_val (word : Word) : (wordDenote word).val = word.val := by
  exact Eq.trans (wordRing.ofNat_rep word.val) (modEqOfLt word.isLt)

theorem wordDenote_eq (word : Word) : wordDenote word = word := by
  apply Fin.ext
  exact wordDenote_eq_val word

theorem wordDenote_ofNat (value : Nat) :
    (wordDenote (Word.ofNat value)).val = value % wordModulus := by
  exact Eq.trans (wordDenote_eq_val (Word.ofNat value)) (Word.ofNat_val value)

theorem wordDenote_add_rep (left right : Word) :
    (wordDenote (Word.add left right)).val =
      (left.val + right.val) % wordModulus := by
  exact Eq.trans (wordDenote_eq_val (Word.add left right))
    (wordRing.add_rep left right)

theorem wordDenote_sub_rep (left right : Word) :
    (wordDenote (Word.sub left right)).val =
      (left.val + wordModulus - right.val) % wordModulus := by
  exact Eq.trans (wordDenote_eq_val (Word.sub left right))
    (wordRing.sub_rep left right)

theorem wordDenote_mul_rep (left right : Word) :
    (wordDenote (Word.mul left right)).val =
      (left.val * right.val) % wordModulus := by
  exact Eq.trans (wordDenote_eq_val (Word.mul left right))
    (wordRing.mul_rep left right)

theorem wordDenote_add (left right : Word) :
    wordDenote (Word.add left right) = wordRing.add (wordDenote left) (wordDenote right) := by
  rw [wordDenote_eq, wordDenote_eq, wordDenote_eq]
  exact rfl

theorem wordDenote_sub (left right : Word) :
    wordDenote (Word.sub left right) = wordRing.sub (wordDenote left) (wordDenote right) := by
  rw [wordDenote_eq, wordDenote_eq, wordDenote_eq]
  exact rfl

theorem wordDenote_mul (left right : Word) :
    wordDenote (Word.mul left right) = wordRing.mul (wordDenote left) (wordDenote right) := by
  rw [wordDenote_eq, wordDenote_eq, wordDenote_eq]
  exact rfl

theorem wordDenote_bitAnd (left right : Word) :
    wordDenote (Word.bitAnd left right) =
      wordRing.ofNat (Word.bitwise Bool.and wordBits left.val right.val) := by
  rw [wordDenote_eq]
  exact (wordRing.ofNat_agrees _).symm

theorem wordDenote_tableSpread (key : Word) :
    wordDenote (Word.tableSpread key) =
      wordRing.ofNat (Word.bitwise Bool.and wordBits
        (Word.tableSpreadProduct key).val Word.spreadMask.val) := by
  exact wordDenote_bitAnd (Word.tableSpreadProduct key) Word.spreadMask

def cellDenote (cell : Word × Word) :
    Apeiron.Trichotomy.AnyRing.RingCell Word :=
  (wordDenote cell.1, wordDenote cell.2)

def cellMul (sigma : Word) (left right : Word × Word) : Word × Word :=
  (Word.add (Word.mul left.1 right.1) (Word.mul (Word.mul sigma left.2) right.2),
    Word.add (Word.mul left.1 right.2) (Word.mul left.2 right.1))

def cellNormForm (sigma : Word) (cell : Word × Word) : Word :=
  Word.sub (Word.mul cell.1 cell.1) (Word.mul (Word.mul sigma cell.2) cell.2)

theorem cellDenote_mul (sigma : Word) (left right : Word × Word) :
    cellDenote (cellMul sigma left right) =
      Apeiron.Trichotomy.AnyRing.ringMul (wordDenote sigma)
        (cellDenote left) (cellDenote right) := by
  apply Prod.ext
  · exact wordDenote_add _ _ |>.trans
      (congrArg₂ (· + ·) (wordDenote_mul _ _)
        (Eq.trans (wordDenote_mul _ _)
          (congrArg (fun value => value * wordDenote right.2) (wordDenote_mul _ _))))
  · exact wordDenote_add _ _ |>.trans
      (congrArg₂ (· + ·) (wordDenote_mul _ _) (wordDenote_mul _ _))

theorem cellNormForm_denote (sigma : Word) (cell : Word × Word) :
    wordDenote (cellNormForm sigma cell) =
      Apeiron.Trichotomy.AnyRing.ringNormForm (wordDenote sigma) (cellDenote cell) := by
  unfold cellDenote
  rw [wordDenote_eq (cellNormForm sigma cell), wordDenote_eq sigma,
    wordDenote_eq cell.1, wordDenote_eq cell.2]
  unfold cellNormForm Apeiron.Trichotomy.AnyRing.ringNormForm
  rw [pow_two, pow_two]
  change Word.sub (Word.mul cell.1 cell.1)
      (Word.mul (Word.mul sigma cell.2) cell.2) =
    Word.sub (Word.mul cell.1 cell.1)
      (Word.mul sigma (Word.mul cell.2 cell.2))
  exact congrArg (Word.sub (Word.mul cell.1 cell.1))
    (wordRing.mul_assoc sigma cell.2 cell.2)

theorem wordMul_comm_denote (left right : Word) :
    wordDenote (Word.mul left right) = wordDenote (Word.mul right left) := by
  rw [wordDenote_eq, wordDenote_eq]
  exact wordRing.mul_comm left right

theorem wordMul_assoc_denote (left middle right : Word) :
    wordDenote (Word.mul (Word.mul left middle) right) =
      wordDenote (Word.mul left (Word.mul middle right)) := by
  rw [wordDenote_eq, wordDenote_eq]
  exact wordRing.mul_assoc left middle right

private theorem ringMul_comm_word (sigma : Word) (left right : Word × Word) :
    Apeiron.Trichotomy.AnyRing.ringMul sigma left right =
      Apeiron.Trichotomy.AnyRing.ringMul sigma right left := by
  apply Prod.ext
  · change left.1 * right.1 + sigma * left.2 * right.2 =
      right.1 * left.1 + sigma * right.2 * left.2
    change Word.add (Word.mul left.1 right.1) (Word.mul (Word.mul sigma left.2) right.2) =
      Word.add (Word.mul right.1 left.1) (Word.mul (Word.mul sigma right.2) left.2)
    rw [wordMul_comm left.1 right.1, wordMul_assoc sigma left.2 right.2,
      wordMul_comm left.2 right.2, ← wordMul_assoc sigma right.2 left.2]
  · change left.1 * right.2 + left.2 * right.1 =
      right.1 * left.2 + right.2 * left.1
    change Word.add (Word.mul left.1 right.2) (Word.mul left.2 right.1) =
      Word.add (Word.mul right.1 left.2) (Word.mul right.2 left.1)
    rw [wordMul_comm left.1 right.2, wordMul_comm left.2 right.1,
      wordAdd_comm]

private theorem addFourRotate (first second third fourth : Word) :
    Word.add (Word.add first second) (Word.add third fourth) =
      Word.add (Word.add first third) (Word.add fourth second) := by
  rw [wordAdd_assoc first second, wordAdd_assoc first third]
  exact congrArg (Word.add first) (Eq.trans
    (wordAdd_assoc second third fourth).symm
    (Eq.trans (congrArg (fun value => Word.add value fourth) (wordAdd_comm second third))
      (Eq.trans (wordAdd_assoc third second fourth)
        (congrArg (Word.add third) (wordAdd_comm second fourth)))))

private theorem ringMul_assoc_word (sigma : Word)
    (left middle right : Word × Word) :
    Apeiron.Trichotomy.AnyRing.ringMul sigma
        (Apeiron.Trichotomy.AnyRing.ringMul sigma left middle) right =
      Apeiron.Trichotomy.AnyRing.ringMul sigma left
        (Apeiron.Trichotomy.AnyRing.ringMul sigma middle right) := by
  apply Prod.ext
  · change (left.1 * middle.1 + sigma * left.2 * middle.2) * right.1 +
      sigma * (left.1 * middle.2 + left.2 * middle.1) * right.2 =
      left.1 * (middle.1 * right.1 + sigma * middle.2 * right.2) +
      sigma * left.2 * (middle.1 * right.2 + middle.2 * right.1)
    change Word.add
      (Word.mul (Word.add (Word.mul left.1 middle.1)
        (Word.mul (Word.mul sigma left.2) middle.2)) right.1)
      (Word.mul (Word.mul sigma (Word.add (Word.mul left.1 middle.2)
        (Word.mul left.2 middle.1))) right.2) =
      Word.add
        (Word.mul left.1 (Word.add (Word.mul middle.1 right.1)
          (Word.mul (Word.mul sigma middle.2) right.2)))
        (Word.mul (Word.mul sigma left.2) (Word.add (Word.mul middle.1 right.2)
          (Word.mul middle.2 right.1)))
    rw [wordAdd_mul, wordMul_add, wordMul_add, wordAdd_mul, wordMul_add]
    rw [wordMul_assoc, wordMul_assoc, wordMul_assoc, wordMul_assoc,
      wordMul_assoc, wordMul_assoc, wordMul_assoc, wordMul_assoc]
    rw [← wordMul_assoc sigma left.2 (Word.mul middle.2 right.1),
      ← wordMul_assoc sigma left.1 (Word.mul middle.2 right.2),
      wordMul_comm sigma left.1,
      wordMul_assoc left.1 sigma (Word.mul middle.2 right.2),
      ← wordMul_assoc sigma left.2 (Word.mul middle.1 right.2)]
    exact addFourRotate _ _ _ _
  · change (left.1 * middle.1 + sigma * left.2 * middle.2) * right.2 +
      (left.1 * middle.2 + left.2 * middle.1) * right.1 =
      left.1 * (middle.1 * right.2 + middle.2 * right.1) +
      left.2 * (middle.1 * right.1 + sigma * middle.2 * right.2)
    change Word.add
      (Word.mul (Word.add (Word.mul left.1 middle.1)
        (Word.mul (Word.mul sigma left.2) middle.2)) right.2)
      (Word.mul (Word.add (Word.mul left.1 middle.2)
        (Word.mul left.2 middle.1)) right.1) =
      Word.add
        (Word.mul left.1 (Word.add (Word.mul middle.1 right.2)
          (Word.mul middle.2 right.1)))
        (Word.mul left.2 (Word.add (Word.mul middle.1 right.1)
          (Word.mul (Word.mul sigma middle.2) right.2)))
    rw [wordAdd_mul, wordAdd_mul, wordMul_add, wordMul_add]
    rw [wordMul_assoc, wordMul_assoc, wordMul_assoc,
      wordMul_assoc, wordMul_assoc, wordMul_assoc]
    rw [← wordMul_assoc sigma left.2 (Word.mul middle.2 right.2),
      wordMul_comm sigma left.2,
      wordMul_assoc left.2 sigma (Word.mul middle.2 right.2)]
    exact addFourRotate _ _ _ _

theorem cellMul_comm_denote (sigma : Word) (left right : Word × Word) :
    cellDenote (cellMul sigma left right) = cellDenote (cellMul sigma right left) := by
  rw [cellDenote_mul, cellDenote_mul]
  exact ringMul_comm_word (wordDenote sigma) (cellDenote left) (cellDenote right)

theorem cellMul_assoc_denote (sigma : Word) (left middle right : Word × Word) :
    cellDenote (cellMul sigma (cellMul sigma left middle) right) =
      cellDenote (cellMul sigma left (cellMul sigma middle right)) := by
  rw [cellDenote_mul, cellDenote_mul, cellDenote_mul, cellDenote_mul]
  exact ringMul_assoc_word (wordDenote sigma)
    (cellDenote left) (cellDenote middle) (cellDenote right)

private theorem crossCancelIdentity {RingCarrier : Type} [Add RingCarrier] [Neg RingCarrier]
    [Sub RingCarrier] [Zero RingCarrier]
    (assoc : ∀ (a b c : RingCarrier), a + b + c = a + (b + c))
    (comm : ∀ (left right : RingCarrier), left + right = right + left)
    (subEq : ∀ (left right : RingCarrier), left - right = left + -right)
    (negAdd : ∀ (left right : RingCarrier), -(left + right) = -right + -left)
    (negNeg : ∀ (value : RingCarrier), - -value = value)
    (addNeg : ∀ (value : RingCarrier), value + -value = 0)
    (zeroAdd : ∀ (value : RingCarrier), 0 + value = value)
    (a cross d b c : RingCarrier) :
    a + cross + d - (b + cross + c) = a - c - (b - d) := by
  have leftComm (left middle right : RingCarrier) :
      left + (middle + right) = middle + (left + right) :=
    Eq.trans (assoc left middle right).symm
      (Eq.trans (congrArg (fun value => value + right) (comm left middle))
        (assoc middle left right))
  rw [subEq, subEq, subEq, negAdd, negAdd]
  rw [subEq b d, negAdd, negNeg]
  rw [assoc (a + cross) d (-c + (-cross + -b)), assoc a cross,
    leftComm d (-c) (-cross + -b),
    leftComm d (-cross) (-b),
    leftComm (-c) (-cross) (d + -b),
    ← assoc cross (-cross), addNeg, zeroAdd]
  exact (assoc a (-c) (d + -b)).symm

private theorem squareAddLocal {RingCarrier : Type} [CommRing RingCarrier]
    (left right : RingCarrier) :
    (left + right) ^ 2 = left * left + left * right + (left * right + right * right) := by
  rw [pow_two, add_mul, mul_add, mul_add, mul_comm right left, add_assoc]

private theorem mulFourSwap {RingCarrier : Type} [CommRing RingCarrier]
    (a b c d : RingCarrier) : (a * b) * (c * d) = (a * c) * (b * d) := by
  rw [mul_assoc, mul_left_comm b c, ← mul_assoc]

private theorem mulCross {RingCarrier : Type} [CommRing RingCarrier]
    (a c sigma b d : RingCarrier) :
    (a * c) * ((sigma * b) * d) = sigma * ((a * d) * (b * c)) := by
  calc
    (a * c) * ((sigma * b) * d) = a * (c * (sigma * (b * d))) := by
      rw [mul_assoc sigma b d, mul_assoc a c]
    _ = a * (sigma * (c * (b * d))) :=
      congrArg (fun value => a * value) (mul_left_comm c sigma (b * d))
    _ = sigma * (a * (c * (b * d))) := mul_left_comm a sigma _
    _ = sigma * (a * (b * (c * d))) := by rw [mul_left_comm c b]
    _ = sigma * ((a * b) * (c * d)) := by rw [mul_assoc a b]
    _ = sigma * ((a * b) * (d * c)) := by rw [mul_comm c d]
    _ = sigma * ((a * d) * (b * c)) :=
      congrArg (fun value => sigma * value) (mulFourSwap a b d c)

private theorem mulSigmaSquare {RingCarrier : Type} [CommRing RingCarrier]
    (sigma value other : RingCarrier) :
    sigma * ((value * other) * (value * other)) =
      (sigma * (value * value)) * (other * other) := by
  exact Eq.trans
    (congrArg (fun product => sigma * product) (mulFourSwap value other value other))
    (mul_assoc sigma (value * value) (other * other)).symm

private theorem mulThreeSquare {RingCarrier : Type} [CommRing RingCarrier]
    (sigma value other : RingCarrier) :
    ((sigma * value) * other) * ((sigma * value) * other) =
      (sigma * (value * value)) * (sigma * (other * other)) := by
  calc
    ((sigma * value) * other) * ((sigma * value) * other) =
        ((sigma * value) * (sigma * value)) * (other * other) :=
      mulFourSwap (sigma * value) other (sigma * value) other
    _ = ((sigma * sigma) * (value * value)) * (other * other) :=
      congrArg (fun product => product * (other * other))
        (mulFourSwap sigma value sigma value)
    _ = (sigma * sigma) * ((value * value) * (other * other)) :=
      mul_assoc _ _ _
    _ = (sigma * (value * value)) * (sigma * (other * other)) :=
      mulFourSwap sigma sigma (value * value) (other * other)

private theorem ringNormForm_mul_local {RingCarrier : Type} [CommRing RingCarrier]
    (sigma : RingCarrier) (left right : RingCarrier × RingCarrier) :
    Apeiron.Trichotomy.AnyRing.ringNormForm sigma
        (Apeiron.Trichotomy.AnyRing.ringMul sigma left right) =
      Apeiron.Trichotomy.AnyRing.ringNormForm sigma left *
        Apeiron.Trichotomy.AnyRing.ringNormForm sigma right := by
  unfold Apeiron.Trichotomy.AnyRing.ringNormForm Apeiron.Trichotomy.AnyRing.ringMul
  rw [squareAddLocal, squareAddLocal]
  rw [mul_sub, sub_mul, mul_add, sub_mul, mul_add, mul_add]
  repeat rw [pow_two]
  have firstSquare : left.1 * right.1 * (left.1 * right.1) =
      left.1 * left.1 * (right.1 * right.1) :=
    mulFourSwap left.1 right.1 left.1 right.1
  have crossTerm : (left.1 * right.1) * ((sigma * left.2) * right.2) =
      sigma * ((left.1 * right.2) * (left.2 * right.1)) :=
    mulCross left.1 right.1 sigma left.2 right.2
  have sigmaSquare : sigma * left.2 * right.2 * (sigma * left.2 * right.2) =
      sigma * (left.2 * left.2) * (sigma * (right.2 * right.2)) :=
    mulThreeSquare sigma left.2 right.2
  have rightSquare : sigma * (left.1 * right.2 * (left.1 * right.2)) =
      left.1 * left.1 * (sigma * (right.2 * right.2)) := by
    rw [mulFourSwap left.1 right.2 left.1 right.2, mul_left_comm sigma]
  have leftSquare : sigma * (left.2 * right.1 * (left.2 * right.1)) =
      sigma * (left.2 * left.2) * (right.1 * right.1) :=
    mulSigmaSquare sigma left.2 right.1
  rw [firstSquare, crossTerm, sigmaSquare, rightSquare, leftSquare]
  let a := left.1 * left.1 * (right.1 * right.1)
  let cross := sigma * ((left.1 * right.2) * (left.2 * right.1))
  let d := sigma * (left.2 * left.2) * (sigma * (right.2 * right.2))
  let b := left.1 * left.1 * (sigma * (right.2 * right.2))
  let c := sigma * (left.2 * left.2) * (right.1 * right.1)
  change a + cross + (cross + d) - (b + cross + (cross + c)) =
    a - c - (b - d)
  rw [add_assoc a cross (cross + d), ← add_assoc cross cross d,
    ← add_assoc a (cross + cross) d,
    add_assoc b cross (cross + c), ← add_assoc cross cross c,
    ← add_assoc b (cross + cross) c]
  exact crossCancelIdentity add_assoc add_comm sub_eq_add_neg neg_add_rev
    neg_neg add_neg_cancel zero_add a (cross + cross) d b c

private theorem ringNormForm_mul_word (sigma : Word) (left right : Word × Word) :
    Apeiron.Trichotomy.AnyRing.ringNormForm sigma
        (Apeiron.Trichotomy.AnyRing.ringMul sigma left right) =
      Apeiron.Trichotomy.AnyRing.ringNormForm sigma left *
        Apeiron.Trichotomy.AnyRing.ringNormForm sigma right :=
  ringNormForm_mul_local sigma left right

theorem cellNormForm_mul_denote (sigma : Word) (left right : Word × Word) :
    wordDenote (cellNormForm sigma (cellMul sigma left right)) =
      wordRing.mul (wordDenote (cellNormForm sigma left))
        (wordDenote (cellNormForm sigma right)) := by
  rw [cellNormForm_denote, cellDenote_mul,
    cellNormForm_denote, cellNormForm_denote]
  exact ringNormForm_mul_word (wordDenote sigma) (cellDenote left) (cellDenote right)

end Slang.Refinement
