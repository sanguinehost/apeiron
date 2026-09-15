import Slang.Word
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Word.Audit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Word.Audit

#audit_empty Slang.Word.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.wordBits : Nat
#audit_empty Slang.wordModulus : Nat
#audit_empty Slang.wordSignBit : Nat
#audit_empty Slang.wordModulus_pos : 0 < Slang.wordModulus
#audit_empty Slang.Word : Type
#audit_empty Slang.Word.ofNat : Nat → Slang.Word
#audit_empty Slang.Word.bits : Slang.Word → Nat
#audit_empty Slang.Word.signed : Slang.Word → Int
#audit_empty Slang.Word.add : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.sub : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.mul : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.bitwise : (Bool → Bool → Bool) → Nat → Nat → Nat → Nat
#audit_empty Slang.Word.bitAnd : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.bitOr : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.bitXor : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.decidableEq : DecidableEq Slang.Word
#audit_empty Slang.Word.instDecidableEq : DecidableEq Slang.Word
#audit_empty Slang.Word.beq : Slang.Word → Slang.Word → Bool
#audit_empty Slang.Word.instBEq : BEq Slang.Word
#audit_empty Slang.Word.instInhabited : Inhabited Slang.Word
#audit_empty Slang.Word.bool : Bool → Slang.Word
#audit_empty Slang.Word.eq : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.ne : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.lt : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.le : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.gt : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.ge : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Word.nonzero : Slang.Word → Prop
#audit_empty Slang.Word.tablePhi : Slang.Word
#audit_empty Slang.Word.spreadMask : Slang.Word
#audit_empty Slang.Word.spreadPosition : Slang.Word → Slang.Word
#audit_empty Slang.Word.tableSpreadProduct : Slang.Word → Slang.Word
#audit_empty Slang.Word.tableSpread : Slang.Word → Slang.Word
#audit_empty Slang.Word.ofNat_val : ∀ (value : Nat), (Slang.Word.ofNat value).val = value % Slang.wordModulus
#audit_empty Slang.Word.bits_lt_modulus : ∀ (value : Slang.Word), Slang.Word.bits value < Slang.wordModulus
#audit_empty Slang.Word.signed_view : ∀ (value : Slang.Word), Slang.Word.signed value = if value.val < Slang.wordSignBit then Int.ofNat value.val else Int.ofNat value.val - Int.ofNat Slang.wordModulus
#audit_empty Slang.Word.wrapping_add_agreement : ∀ (left right : Slang.Word), (Slang.Word.add left right).val = (left.val + right.val) % Slang.wordModulus
#audit_empty Slang.Word.wrapping_sub_agreement : ∀ (left right : Slang.Word), (Slang.Word.sub left right).val = (left.val + Slang.wordModulus - right.val) % Slang.wordModulus
#audit_empty Slang.Word.wrapping_mul_agreement : ∀ (left right : Slang.Word), (Slang.Word.mul left right).val = (left.val * right.val) % Slang.wordModulus
#audit_empty Slang.Word.bitAnd_agreement : ∀ (left right : Slang.Word), (Slang.Word.bitAnd left right).val = Slang.Word.bitwise Bool.and Slang.wordBits left.val right.val % Slang.wordModulus
#audit_empty Slang.Word.bitOr_agreement : ∀ (left right : Slang.Word), (Slang.Word.bitOr left right).val = Slang.Word.bitwise Bool.or Slang.wordBits left.val right.val % Slang.wordModulus
#audit_empty Slang.Word.bitXor_agreement : ∀ (left right : Slang.Word), (Slang.Word.bitXor left right).val = Slang.Word.bitwise Bool.xor Slang.wordBits left.val right.val % Slang.wordModulus
#audit_empty Slang.Word.signed_zero : Slang.Word.signed (Slang.Word.ofNat 0) = 0
#audit_empty Slang.Word.signed_allOnes : Slang.Word.signed (Slang.Word.ofNat (Slang.wordModulus - 1)) = -1
#audit_empty Slang.Word.tableSpread_multiplies_before_mask : ∀ (key : Slang.Word), Slang.Word.tableSpread key = Slang.Word.bitAnd (Slang.Word.mul (Slang.Word.spreadPosition key) Slang.Word.tablePhi) Slang.Word.spreadMask
#audit_empty Slang.Word.tableSpread_is_word_bounded : ∀ (position : Slang.Word), Slang.Word.bits (Slang.Word.tableSpread position) < Slang.wordModulus
#audit_empty Slang.Word.wrapping_add_boundary : Slang.Word.add (Slang.Word.ofNat (Slang.wordModulus - 1)) (Slang.Word.ofNat 1) = Slang.Word.ofNat 0
#audit_empty Slang.Word.wrapping_mul_boundary : Slang.Word.mul (Slang.Word.ofNat Slang.wordSignBit) (Slang.Word.ofNat 2) = Slang.Word.ofNat 0
#audit_empty Slang.Word.signed_comparison_fixture : Slang.Word.lt (Slang.Word.ofNat (Slang.wordModulus - 1)) (Slang.Word.ofNat 0) = Slang.Word.ofNat 1
#audit_empty Slang.Word.bitwise_fixture : Slang.Word.bitAnd (Slang.Word.ofNat 0xF0) (Slang.Word.ofNat 0x3C) = Slang.Word.ofNat 0x30 ∧ Slang.Word.bitOr (Slang.Word.ofNat 0xF0) (Slang.Word.ofNat 0x3C) = Slang.Word.ofNat 0xFC ∧ Slang.Word.bitXor (Slang.Word.ofNat 0xF0) (Slang.Word.ofNat 0x3C) = Slang.Word.ofNat 0xCC
