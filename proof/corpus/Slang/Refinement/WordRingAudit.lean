import Slang.Refinement.WordRing
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Refinement.WordRingAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Refinement.WordRingAudit

#audit_empty Slang.Refinement.WordRingAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Refinement.addSubCancel : ∀ (left right : Nat), left + right - right = left
#audit_empty Slang.Refinement.subAddCancel : ∀ {left right : Nat}, right ≤ left → left - right + right = left
#audit_empty Slang.Refinement.addSubAssoc : ∀ {middle subtrahend : Nat}, subtrahend ≤ middle → ∀ (left : Nat), left + middle - subtrahend = left + (middle - subtrahend)
#audit_empty Slang.Refinement.addSubOfLe : ∀ {left right : Nat}, left ≤ right → left + (right - left) = right
#audit_empty Slang.Refinement.mulAssoc : ∀ (left middle right : Nat), left * middle * right = left * (middle * right)
#audit_empty Slang.Refinement.wordModulusOneLt : 1 < Slang.wordModulus

#audit_empty Slang.Refinement.modEqOfLt : ∀ {value modulus : Nat}, value < modulus → value % modulus = value
#audit_empty Slang.Refinement.modCoreGoFuelCongr : ∀ (value modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus) (bound₁ : value < fuel₁) (bound₂ : value < fuel₂), Nat.modCore.go modulus positive fuel₁ value bound₁ = Nat.modCore.go modulus positive fuel₂ value bound₂
#audit_empty Slang.Refinement.modCoreGoValueCongr : ∀ (left right modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus) (leftBound : left < fuel₁) (rightBound : right < fuel₂), left = right → Nat.modCore.go modulus positive fuel₁ left leftBound = Nat.modCore.go modulus positive fuel₂ right rightBound
#audit_empty Slang.Refinement.modCoreGoStep : ∀ (value modulus fuel : Nat) (positive : 0 < modulus) (bound : value < fuel + 1), Nat.modCore.go modulus positive (fuel + 1) value bound = if lower : modulus ≤ value then Nat.modCore.go modulus positive fuel (value - modulus) (Nat.div_rec_fuel_lemma positive lower bound) else value
#audit_empty Slang.Refinement.modCorePositive : ∀ (value modulus : Nat) (positive : 0 < modulus), Nat.modCore value modulus = Nat.modCore.go modulus positive (value + 1) value (Nat.lt_succ_self value)
#audit_empty Slang.Refinement.modCoreEqMod : ∀ (value modulus : Nat), Nat.modCore value modulus = value % modulus
#audit_empty Slang.Refinement.modZero : ∀ (value : Nat), value % 0 = value
#audit_empty Slang.Refinement.modCoreAddModulus : ∀ (value modulus : Nat), 0 < modulus → Nat.modCore (value + modulus) modulus = Nat.modCore value modulus
#audit_empty Slang.Refinement.modAddModulus : ∀ (value modulus : Nat), (value + modulus) % modulus = value % modulus
#audit_empty Slang.Refinement.modEqSub : ∀ {value modulus : Nat}, 0 < modulus → modulus ≤ value → value % modulus = (value - modulus) % modulus
#audit_empty Slang.Refinement.modAddAbsorbLeft : ∀ (left right modulus : Nat), (left % modulus + right) % modulus = (left + right) % modulus
#audit_empty Slang.Refinement.modAddAbsorbRight : ∀ (left right modulus : Nat), (left + right % modulus) % modulus = (left + right) % modulus
#audit_empty Slang.Refinement.modMulAbsorbRight : ∀ (left right modulus : Nat), (left * (right % modulus)) % modulus = (left * right) % modulus
#audit_empty Slang.Refinement.modMulAbsorbLeft : ∀ (left right modulus : Nat), ((left % modulus) * right) % modulus = (left * right) % modulus

#audit_empty Slang.Refinement.wordAdd_comm : ∀ (left right : Slang.Word), Slang.Word.add left right = Slang.Word.add right left
#audit_empty Slang.Refinement.wordAdd_assoc : ∀ (left middle right : Slang.Word), Slang.Word.add (Slang.Word.add left middle) right = Slang.Word.add left (Slang.Word.add middle right)
#audit_empty Slang.Refinement.wordAdd_zero_right : ∀ (value : Slang.Word), Slang.Word.add value (Slang.Word.ofNat 0) = value
#audit_empty Slang.Refinement.wordAdd_zero_left : ∀ (value : Slang.Word), Slang.Word.add (Slang.Word.ofNat 0) value = value
#audit_empty Slang.Refinement.wordAdd_neg_right : ∀ (value : Slang.Word), Slang.Word.add value (Slang.Word.sub (Slang.Word.ofNat 0) value) = Slang.Word.ofNat 0
#audit_empty Slang.Refinement.wordSub_eq_add_neg : ∀ (left right : Slang.Word), Slang.Word.sub left right = Slang.Word.add left (Slang.Word.sub (Slang.Word.ofNat 0) right)
#audit_empty Slang.Refinement.wordMul_comm : ∀ (left right : Slang.Word), Slang.Word.mul left right = Slang.Word.mul right left
#audit_empty Slang.Refinement.wordMul_assoc : ∀ (left middle right : Slang.Word), Slang.Word.mul (Slang.Word.mul left middle) right = Slang.Word.mul left (Slang.Word.mul middle right)
#audit_empty Slang.Refinement.wordMul_one_right : ∀ (value : Slang.Word), Slang.Word.mul value (Slang.Word.ofNat 1) = value
#audit_empty Slang.Refinement.wordMul_one_left : ∀ (value : Slang.Word), Slang.Word.mul (Slang.Word.ofNat 1) value = value
#audit_empty Slang.Refinement.wordMul_add : ∀ (left middle right : Slang.Word), Slang.Word.mul left (Slang.Word.add middle right) = Slang.Word.add (Slang.Word.mul left middle) (Slang.Word.mul left right)
#audit_empty Slang.Refinement.wordAdd_mul : ∀ (left middle right : Slang.Word), Slang.Word.mul (Slang.Word.add left middle) right = Slang.Word.add (Slang.Word.mul left right) (Slang.Word.mul middle right)
#audit_empty Slang.Refinement.WordRingLaws : Type
#audit_empty Slang.Refinement.wordRing : Slang.Refinement.WordRingLaws
