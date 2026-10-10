import Iota.Refinement.WordRing
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Refinement.WordRingAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Refinement.WordRingAudit

#audit_empty Iota.Refinement.WordRingAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Refinement.addSubCancel : ∀ (left right : Nat), left + right - right = left
#audit_empty Iota.Refinement.subAddCancel : ∀ {left right : Nat}, right ≤ left → left - right + right = left
#audit_empty Iota.Refinement.addSubAssoc : ∀ {middle subtrahend : Nat}, subtrahend ≤ middle → ∀ (left : Nat), left + middle - subtrahend = left + (middle - subtrahend)
#audit_empty Iota.Refinement.addSubOfLe : ∀ {left right : Nat}, left ≤ right → left + (right - left) = right
#audit_empty Iota.Refinement.mulAssoc : ∀ (left middle right : Nat), left * middle * right = left * (middle * right)
#audit_empty Iota.Refinement.wordModulusOneLt : 1 < Iota.wordModulus

#audit_empty Iota.Refinement.modEqOfLt : ∀ {value modulus : Nat}, value < modulus → value % modulus = value
#audit_empty Iota.Refinement.modCoreGoFuelCongr : ∀ (value modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus) (bound₁ : value < fuel₁) (bound₂ : value < fuel₂), Nat.modCore.go modulus positive fuel₁ value bound₁ = Nat.modCore.go modulus positive fuel₂ value bound₂
#audit_empty Iota.Refinement.modCoreGoValueCongr : ∀ (left right modulus fuel₁ fuel₂ : Nat) (positive : 0 < modulus) (leftBound : left < fuel₁) (rightBound : right < fuel₂), left = right → Nat.modCore.go modulus positive fuel₁ left leftBound = Nat.modCore.go modulus positive fuel₂ right rightBound
#audit_empty Iota.Refinement.modCoreGoStep : ∀ (value modulus fuel : Nat) (positive : 0 < modulus) (bound : value < fuel + 1), Nat.modCore.go modulus positive (fuel + 1) value bound = if lower : modulus ≤ value then Nat.modCore.go modulus positive fuel (value - modulus) (Nat.div_rec_fuel_lemma positive lower bound) else value
#audit_empty Iota.Refinement.modCorePositive : ∀ (value modulus : Nat) (positive : 0 < modulus), Nat.modCore value modulus = Nat.modCore.go modulus positive (value + 1) value (Nat.lt_succ_self value)
#audit_empty Iota.Refinement.modCoreEqMod : ∀ (value modulus : Nat), Nat.modCore value modulus = value % modulus
#audit_empty Iota.Refinement.modZero : ∀ (value : Nat), value % 0 = value
#audit_empty Iota.Refinement.modCoreAddModulus : ∀ (value modulus : Nat), 0 < modulus → Nat.modCore (value + modulus) modulus = Nat.modCore value modulus
#audit_empty Iota.Refinement.modAddModulus : ∀ (value modulus : Nat), (value + modulus) % modulus = value % modulus
#audit_empty Iota.Refinement.modEqSub : ∀ {value modulus : Nat}, 0 < modulus → modulus ≤ value → value % modulus = (value - modulus) % modulus
#audit_empty Iota.Refinement.modAddAbsorbLeft : ∀ (left right modulus : Nat), (left % modulus + right) % modulus = (left + right) % modulus
#audit_empty Iota.Refinement.modAddAbsorbRight : ∀ (left right modulus : Nat), (left + right % modulus) % modulus = (left + right) % modulus
#audit_empty Iota.Refinement.modMulAbsorbRight : ∀ (left right modulus : Nat), (left * (right % modulus)) % modulus = (left * right) % modulus
#audit_empty Iota.Refinement.modMulAbsorbLeft : ∀ (left right modulus : Nat), ((left % modulus) * right) % modulus = (left * right) % modulus

#audit_empty Iota.Refinement.wordAdd_comm : ∀ (left right : Iota.Word), Iota.Word.add left right = Iota.Word.add right left
#audit_empty Iota.Refinement.wordAdd_assoc : ∀ (left middle right : Iota.Word), Iota.Word.add (Iota.Word.add left middle) right = Iota.Word.add left (Iota.Word.add middle right)
#audit_empty Iota.Refinement.wordAdd_zero_right : ∀ (value : Iota.Word), Iota.Word.add value (Iota.Word.ofNat 0) = value
#audit_empty Iota.Refinement.wordAdd_zero_left : ∀ (value : Iota.Word), Iota.Word.add (Iota.Word.ofNat 0) value = value
#audit_empty Iota.Refinement.wordAdd_neg_right : ∀ (value : Iota.Word), Iota.Word.add value (Iota.Word.sub (Iota.Word.ofNat 0) value) = Iota.Word.ofNat 0
#audit_empty Iota.Refinement.wordSub_eq_add_neg : ∀ (left right : Iota.Word), Iota.Word.sub left right = Iota.Word.add left (Iota.Word.sub (Iota.Word.ofNat 0) right)
#audit_empty Iota.Refinement.wordMul_comm : ∀ (left right : Iota.Word), Iota.Word.mul left right = Iota.Word.mul right left
#audit_empty Iota.Refinement.wordMul_assoc : ∀ (left middle right : Iota.Word), Iota.Word.mul (Iota.Word.mul left middle) right = Iota.Word.mul left (Iota.Word.mul middle right)
#audit_empty Iota.Refinement.wordMul_one_right : ∀ (value : Iota.Word), Iota.Word.mul value (Iota.Word.ofNat 1) = value
#audit_empty Iota.Refinement.wordMul_one_left : ∀ (value : Iota.Word), Iota.Word.mul (Iota.Word.ofNat 1) value = value
#audit_empty Iota.Refinement.wordMul_add : ∀ (left middle right : Iota.Word), Iota.Word.mul left (Iota.Word.add middle right) = Iota.Word.add (Iota.Word.mul left middle) (Iota.Word.mul left right)
#audit_empty Iota.Refinement.wordAdd_mul : ∀ (left middle right : Iota.Word), Iota.Word.mul (Iota.Word.add left middle) right = Iota.Word.add (Iota.Word.mul left right) (Iota.Word.mul middle right)
#audit_empty Iota.Refinement.WordRingLaws : Type
#audit_empty Iota.Refinement.wordRing : Iota.Refinement.WordRingLaws
