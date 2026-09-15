import Slang.Word

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Refinement

open Slang

theorem addSubCancel (left right : Nat) : left + right - right = left := by
  induction right with
  | zero => rfl
  | succ right ih =>
      rw [Nat.add_succ, Nat.succ_sub_succ_eq_sub]
      exact ih

theorem subAddCancel {left right : Nat} (bound : right ≤ left) :
    left - right + right = left := by
  induction left generalizing right with
  | zero =>
      have equal : right = 0 := Nat.eq_zero_of_le_zero bound
      cases equal
      rfl
  | succ left ih =>
      cases right with
      | zero => rfl
      | succ right =>
          rw [Nat.succ_sub_succ_eq_sub, Nat.add_succ]
          exact congrArg Nat.succ (ih (Nat.le_of_succ_le_succ bound))

theorem addSubAssoc {middle subtrahend : Nat} (bound : subtrahend ≤ middle)
    (left : Nat) : left + middle - subtrahend = left + (middle - subtrahend) := by
  induction middle generalizing subtrahend with
  | zero =>
      have equal : subtrahend = 0 := Nat.eq_zero_of_le_zero bound
      cases equal
      rfl
  | succ middle ih =>
      cases subtrahend with
      | zero => rfl
      | succ subtrahend =>
          rw [Nat.add_succ, Nat.succ_sub_succ_eq_sub, Nat.succ_sub_succ_eq_sub]
          exact ih (Nat.le_of_succ_le_succ bound)

theorem addSubOfLe {left right : Nat} (bound : left ≤ right) :
    left + (right - left) = right := by
  rw [Nat.add_comm]
  exact subAddCancel bound

theorem mulAssoc (left middle right : Nat) :
    left * middle * right = left * (middle * right) := by
  induction right with
  | zero => rfl
  | succ right ih =>
      rw [Nat.mul_succ, Nat.mul_succ, Nat.left_distrib, ih]

theorem wordModulusOneLt : 1 < wordModulus := by decide

theorem modEqOfLt {value modulus : Nat} (bound : value < modulus) :
    value % modulus = value := by
  unfold HMod.hMod instHMod Nat.instMod Nat.mod
  cases value with
  | zero => rfl
  | succ value =>
      cases modulus with
      | zero => exact False.elim (Nat.not_lt_zero _ bound)
      | succ modulus => exact if_neg (Nat.not_le_of_gt bound)

theorem modCoreGoFuelCongr (value modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus)
    (bound₁ : value < fuel₁) (bound₂ : value < fuel₂) :
    Nat.modCore.go modulus positive fuel₁ value bound₁ =
      Nat.modCore.go modulus positive fuel₂ value bound₂ := by
  match fuel₁, fuel₂ with
  | 0, _ => exact False.elim (Nat.not_lt_zero _ bound₁)
  | _, 0 => exact False.elim (Nat.not_lt_zero _ bound₂)
  | fuel₁ + 1, fuel₂ + 1 =>
      unfold Nat.modCore.go
      match Nat.decLe modulus value with
      | isTrue lower =>
          exact modCoreGoFuelCongr (value - modulus) modulus fuel₁ fuel₂ positive
            (Nat.div_rec_fuel_lemma positive lower bound₁)
            (Nat.div_rec_fuel_lemma positive lower bound₂)
      | isFalse _ => rfl
termination_by structural fuel₁

theorem modCoreGoValueCongr (left right modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus)
    (leftBound : left < fuel₁) (rightBound : right < fuel₂) (equal : left = right) :
    Nat.modCore.go modulus positive fuel₁ left leftBound =
      Nat.modCore.go modulus positive fuel₂ right rightBound := by
  cases equal
  exact modCoreGoFuelCongr left modulus fuel₁ fuel₂ positive leftBound rightBound

theorem modCoreGoStep (value modulus fuel : Nat) (positive : 0 < modulus)
    (bound : value < fuel + 1) :
    Nat.modCore.go modulus positive (fuel + 1) value bound =
      if lower : modulus ≤ value then
        Nat.modCore.go modulus positive fuel (value - modulus)
          (Nat.div_rec_fuel_lemma positive lower bound)
      else value := rfl

theorem modCorePositive (value modulus : Nat) (positive : 0 < modulus) :
    Nat.modCore value modulus =
      Nat.modCore.go modulus positive (value + 1) value (Nat.lt_succ_self value) := by
  unfold Nat.modCore
  exact dif_pos positive

theorem modCoreEqMod (value modulus : Nat) : Nat.modCore value modulus = value % modulus := by
  cases value with
  | zero =>
      cases modulus with
      | zero =>
          unfold HMod.hMod instHMod Nat.instMod Nat.mod Nat.modCore
          rfl
      | succ modulus =>
          unfold HMod.hMod instHMod Nat.instMod Nat.mod Nat.modCore Nat.modCore.go
          rfl
  | succ value =>
      cases modulus with
      | zero =>
          unfold HMod.hMod instHMod Nat.instMod Nat.mod Nat.modCore
          rfl
      | succ modulus =>
          unfold HMod.hMod instHMod Nat.instMod Nat.mod
          dsimp
          match Nat.decLe (modulus + 1) (value + 1) with
          | isTrue _ => rfl
          | isFalse notLower =>
              unfold Nat.modCore Nat.modCore.go
              rw [dif_pos (Nat.zero_lt_succ modulus), dif_neg notLower, if_neg notLower]

theorem modCoreAddModulus (value modulus : Nat) (positive : 0 < modulus) :
    Nat.modCore (value + modulus) modulus = Nat.modCore value modulus := by
  refine Eq.trans (modCorePositive (value + modulus) modulus positive) ?_
  refine Eq.trans ?_ (modCorePositive value modulus positive).symm
  refine Eq.trans (modCoreGoStep (value + modulus) modulus (value + modulus) positive
    (Nat.lt_succ_self (value + modulus))) ?_
  refine Eq.trans (dif_pos (Nat.le_add_left modulus value)) ?_
  exact modCoreGoValueCongr (value + modulus - modulus) value modulus
    (value + modulus) (value + 1) positive
    (Nat.div_rec_fuel_lemma positive (Nat.le_add_left modulus value)
      (Nat.lt_succ_self (value + modulus)))
    (Nat.lt_succ_self value) (addSubCancel value modulus)

theorem modZero (value : Nat) : value % 0 = value := by
  cases value with
  | zero => rfl
  | succ value =>
      change Nat.mod (value + 1) 0 = value + 1
      unfold Nat.mod
      change Nat.modCore (Nat.succ value) 0 = Nat.succ value
      unfold Nat.modCore
      rw [dif_neg (Nat.not_lt_zero 0)]

theorem modAddModulus (value modulus : Nat) :
    (value + modulus) % modulus = value % modulus := by
  cases modulus with
  | zero => rfl
  | succ modulus =>
      rw [← modCoreEqMod, ← modCoreEqMod]
      exact modCoreAddModulus value (modulus + 1) (Nat.zero_lt_succ modulus)

theorem modEqSub {value modulus : Nat} (_positive : 0 < modulus)
    (bound : modulus ≤ value) : value % modulus = (value - modulus) % modulus := by
  rw [← modAddModulus (value - modulus) modulus, subAddCancel bound]

theorem modAddAbsorbLeft (left right modulus : Nat) :
    (left % modulus + right) % modulus = (left + right) % modulus := by
  induction left, modulus using Nat.mod.inductionOn with
  | base left modulus base =>
      cases modulus with
      | zero =>
          rw [modZero, modZero, modZero]
      | succ modulus =>
          have bound : left < modulus + 1 := Nat.lt_of_not_ge (fun lower =>
            base ⟨Nat.zero_lt_succ modulus, lower⟩)
          rw [modEqOfLt bound]
  | ind left modulus condition ih =>
      rw [modEqSub condition.1 condition.2]
      rw [modEqSub condition.1 (Nat.le_trans condition.2 (Nat.le_add_right left right))]
      rw [Nat.add_comm left right, addSubAssoc condition.2,
        Nat.add_comm right (left - modulus), ih]

theorem modAddAbsorbRight (left right modulus : Nat) :
    (left + right % modulus) % modulus = (left + right) % modulus := by
  rw [Nat.add_comm left, Nat.add_comm left]
  exact modAddAbsorbLeft right left modulus

theorem modMulAbsorbRight (left right modulus : Nat) :
    (left * (right % modulus)) % modulus = (left * right) % modulus := by
  induction left with
  | zero => rw [Nat.zero_mul, Nat.zero_mul]
  | succ left ih =>
      rw [Nat.succ_mul, Nat.succ_mul]
      rw [← modAddAbsorbLeft (left * (right % modulus)) (right % modulus) modulus]
      rw [ih]
      rw [modAddAbsorbLeft, modAddAbsorbRight]

theorem modMulAbsorbLeft (left right modulus : Nat) :
    ((left % modulus) * right) % modulus = (left * right) % modulus := by
  rw [Nat.mul_comm (left % modulus), Nat.mul_comm left]
  exact modMulAbsorbRight right left modulus

theorem wordAdd_comm (left right : Word) :
    Word.add left right = Word.add right left := by
  apply Fin.ext
  exact congrArg (fun value => value % wordModulus) (Nat.add_comm left.val right.val)

theorem wordAdd_assoc (left middle right : Word) :
    Word.add (Word.add left middle) right = Word.add left (Word.add middle right) := by
  apply Fin.ext
  change ((left.val + middle.val) % wordModulus + right.val) % wordModulus =
    (left.val + (middle.val + right.val) % wordModulus) % wordModulus
  rw [modAddAbsorbLeft, modAddAbsorbRight, Nat.add_assoc]

theorem wordAdd_zero_right (value : Word) :
    Word.add value (Word.ofNat 0) = value := by
  apply Fin.ext
  change (value.val + 0) % wordModulus = value.val
  rw [Nat.add_zero, modEqOfLt value.isLt]

theorem wordAdd_zero_left (value : Word) :
    Word.add (Word.ofNat 0) value = value := by
  rw [wordAdd_comm]
  exact wordAdd_zero_right value

theorem wordAdd_neg_right (value : Word) :
    Word.add value (Word.sub (Word.ofNat 0) value) = Word.ofNat 0 := by
  apply Fin.ext
  change (value.val + (wordModulus - value.val) % wordModulus) % wordModulus = 0
  rw [modAddAbsorbRight, addSubOfLe (Nat.le_of_lt value.isLt)]
  exact modAddModulus 0 wordModulus

theorem wordSub_eq_add_neg (left right : Word) :
    Word.sub left right = Word.add left (Word.sub (Word.ofNat 0) right) := by
  apply Fin.ext
  change (left.val + wordModulus - right.val) % wordModulus =
    (left.val + (wordModulus - right.val) % wordModulus) % wordModulus
  rw [modAddAbsorbRight, addSubAssoc (Nat.le_of_lt right.isLt)]

theorem wordMul_comm (left right : Word) :
    Word.mul left right = Word.mul right left := by
  apply Fin.ext
  exact congrArg (fun value => value % wordModulus) (Nat.mul_comm left.val right.val)

theorem wordMul_assoc (left middle right : Word) :
    Word.mul (Word.mul left middle) right = Word.mul left (Word.mul middle right) := by
  apply Fin.ext
  change (((left.val * middle.val) % wordModulus) * right.val) % wordModulus =
    (left.val * ((middle.val * right.val) % wordModulus)) % wordModulus
  rw [modMulAbsorbLeft, modMulAbsorbRight, mulAssoc]

theorem wordMul_one_right (value : Word) :
    Word.mul value (Word.ofNat 1) = value := by
  apply Fin.ext
  change (value.val * (1 % wordModulus)) % wordModulus = value.val
  rw [modEqOfLt wordModulusOneLt, Nat.mul_one, modEqOfLt value.isLt]

theorem wordMul_one_left (value : Word) :
    Word.mul (Word.ofNat 1) value = value := by
  rw [wordMul_comm]
  exact wordMul_one_right value

theorem wordMul_add (left middle right : Word) :
    Word.mul left (Word.add middle right) =
      Word.add (Word.mul left middle) (Word.mul left right) := by
  apply Fin.ext
  change (left.val * ((middle.val + right.val) % wordModulus)) % wordModulus =
    (((left.val * middle.val) % wordModulus) +
      ((left.val * right.val) % wordModulus)) % wordModulus
  rw [modMulAbsorbRight, modAddAbsorbLeft, modAddAbsorbRight, Nat.left_distrib]

theorem wordAdd_mul (left middle right : Word) :
    Word.mul (Word.add left middle) right =
      Word.add (Word.mul left right) (Word.mul middle right) := by
  rw [wordMul_comm (Word.add left middle) right, wordMul_add]
  rw [wordMul_comm right left, wordMul_comm right middle]

/-- Local constructive ring interface; it does not install global algebra instances. -/
structure WordRingLaws where
  zero : Word
  one : Word
  neg : Word → Word
  ofNat : Nat → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  zero_agrees : zero = Word.ofNat 0
  one_agrees : one = Word.ofNat 1
  neg_agrees : ∀ value, neg value = Word.sub (Word.ofNat 0) value
  ofNat_agrees : ∀ value, ofNat value = Word.ofNat value
  add_agrees : ∀ left right, add left right = Word.add left right
  sub_agrees : ∀ left right, sub left right = Word.sub left right
  mul_agrees : ∀ left right, mul left right = Word.mul left right
  ofNat_rep : ∀ value, (ofNat value).val = value % wordModulus
  add_rep : ∀ left right, (add left right).val = (left.val + right.val) % wordModulus
  sub_rep : ∀ left right,
    (sub left right).val = (left.val + wordModulus - right.val) % wordModulus
  mul_rep : ∀ left right, (mul left right).val = (left.val * right.val) % wordModulus
  add_comm : ∀ left right, add left right = add right left
  add_assoc : ∀ left middle right, add (add left middle) right = add left (add middle right)
  add_zero_left : ∀ value, add zero value = value
  add_zero_right : ∀ value, add value zero = value
  add_neg_right : ∀ value, add value (neg value) = zero
  sub_eq_add_neg : ∀ left right, sub left right = add left (neg right)
  mul_comm : ∀ left right, mul left right = mul right left
  mul_assoc : ∀ left middle right, mul (mul left middle) right = mul left (mul middle right)
  mul_one_left : ∀ value, mul one value = value
  mul_one_right : ∀ value, mul value one = value
  mul_add : ∀ left middle right, mul left (add middle right) = add (mul left middle) (mul left right)
  add_mul : ∀ left middle right, mul (add left middle) right = add (mul left right) (mul middle right)

/-- The wrapping source-word operations with their constructive ring laws. -/
def wordRing : WordRingLaws where
  zero := Word.ofNat 0
  one := Word.ofNat 1
  neg := fun value => Word.sub (Word.ofNat 0) value
  ofNat := Word.ofNat
  add := Word.add
  sub := Word.sub
  mul := Word.mul
  zero_agrees := rfl
  one_agrees := rfl
  neg_agrees := fun _ => rfl
  ofNat_agrees := fun _ => rfl
  add_agrees := fun _ _ => rfl
  sub_agrees := fun _ _ => rfl
  mul_agrees := fun _ _ => rfl
  ofNat_rep := Word.ofNat_val
  add_rep := Word.wrapping_add_agreement
  sub_rep := Word.wrapping_sub_agreement
  mul_rep := Word.wrapping_mul_agreement
  add_comm := wordAdd_comm
  add_assoc := wordAdd_assoc
  add_zero_left := wordAdd_zero_left
  add_zero_right := wordAdd_zero_right
  add_neg_right := wordAdd_neg_right
  sub_eq_add_neg := wordSub_eq_add_neg
  mul_comm := wordMul_comm
  mul_assoc := wordMul_assoc
  mul_one_left := wordMul_one_left
  mul_one_right := wordMul_one_right
  mul_add := wordMul_add
  add_mul := wordAdd_mul

end Slang.Refinement
