import Iota.Word
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Word.Audit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Word.Audit

#audit_empty Iota.Word.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.wordBits : Nat
#audit_empty Iota.wordModulus : Nat
#audit_empty Iota.wordSignBit : Nat
#audit_empty Iota.wordModulus_pos : 0 < Iota.wordModulus
#audit_empty Iota.Word : Type
#audit_empty Iota.Word.ofNat : Nat → Iota.Word
#audit_empty Iota.Word.bits : Iota.Word → Nat
#audit_empty Iota.Word.signed : Iota.Word → Int
#audit_empty Iota.Word.add : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.sub : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.mul : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.bitwise : (Bool → Bool → Bool) → Nat → Nat → Nat → Nat
#audit_empty Iota.Word.bitAnd : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.bitOr : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.bitXor : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.decidableEq : DecidableEq Iota.Word
#audit_empty Iota.Word.instDecidableEq : DecidableEq Iota.Word
#audit_empty Iota.Word.beq : Iota.Word → Iota.Word → Bool
#audit_empty Iota.Word.instBEq : BEq Iota.Word
#audit_empty Iota.Word.instInhabited : Inhabited Iota.Word
#audit_empty Iota.Word.bool : Bool → Iota.Word
#audit_empty Iota.Word.eq : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.ne : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.lt : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.le : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.gt : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.ge : Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Word.nonzero : Iota.Word → Prop
#audit_empty Iota.Word.tablePhi : Iota.Word
#audit_empty Iota.Word.spreadMask : Iota.Word
#audit_empty Iota.Word.spreadPosition : Iota.Word → Iota.Word
#audit_empty Iota.Word.tableSpreadProduct : Iota.Word → Iota.Word
#audit_empty Iota.Word.tableSpread : Iota.Word → Iota.Word
#audit_empty Iota.Word.ofNat_val : ∀ (value : Nat), (Iota.Word.ofNat value).val = value % Iota.wordModulus
#audit_empty Iota.Word.bits_lt_modulus : ∀ (value : Iota.Word), Iota.Word.bits value < Iota.wordModulus
#audit_empty Iota.Word.signed_view : ∀ (value : Iota.Word), Iota.Word.signed value = if value.val < Iota.wordSignBit then Int.ofNat value.val else Int.ofNat value.val - Int.ofNat Iota.wordModulus
#audit_empty Iota.Word.wrapping_add_agreement : ∀ (left right : Iota.Word), (Iota.Word.add left right).val = (left.val + right.val) % Iota.wordModulus
#audit_empty Iota.Word.wrapping_sub_agreement : ∀ (left right : Iota.Word), (Iota.Word.sub left right).val = (left.val + Iota.wordModulus - right.val) % Iota.wordModulus
#audit_empty Iota.Word.wrapping_mul_agreement : ∀ (left right : Iota.Word), (Iota.Word.mul left right).val = (left.val * right.val) % Iota.wordModulus
#audit_empty Iota.Word.bitAnd_agreement : ∀ (left right : Iota.Word), (Iota.Word.bitAnd left right).val = Iota.Word.bitwise Bool.and Iota.wordBits left.val right.val % Iota.wordModulus
#audit_empty Iota.Word.bitOr_agreement : ∀ (left right : Iota.Word), (Iota.Word.bitOr left right).val = Iota.Word.bitwise Bool.or Iota.wordBits left.val right.val % Iota.wordModulus
#audit_empty Iota.Word.bitXor_agreement : ∀ (left right : Iota.Word), (Iota.Word.bitXor left right).val = Iota.Word.bitwise Bool.xor Iota.wordBits left.val right.val % Iota.wordModulus
#audit_empty Iota.Word.signed_zero : Iota.Word.signed (Iota.Word.ofNat 0) = 0
#audit_empty Iota.Word.signed_allOnes : Iota.Word.signed (Iota.Word.ofNat (Iota.wordModulus - 1)) = -1
#audit_empty Iota.Word.tableSpread_multiplies_before_mask : ∀ (key : Iota.Word), Iota.Word.tableSpread key = Iota.Word.bitAnd (Iota.Word.mul (Iota.Word.spreadPosition key) Iota.Word.tablePhi) Iota.Word.spreadMask
#audit_empty Iota.Word.tableSpread_is_word_bounded : ∀ (position : Iota.Word), Iota.Word.bits (Iota.Word.tableSpread position) < Iota.wordModulus
#audit_empty Iota.Word.wrapping_add_boundary : Iota.Word.add (Iota.Word.ofNat (Iota.wordModulus - 1)) (Iota.Word.ofNat 1) = Iota.Word.ofNat 0
#audit_empty Iota.Word.wrapping_mul_boundary : Iota.Word.mul (Iota.Word.ofNat Iota.wordSignBit) (Iota.Word.ofNat 2) = Iota.Word.ofNat 0
#audit_empty Iota.Word.signed_comparison_fixture : Iota.Word.lt (Iota.Word.ofNat (Iota.wordModulus - 1)) (Iota.Word.ofNat 0) = Iota.Word.ofNat 1
#audit_empty Iota.Word.bitwise_fixture : Iota.Word.bitAnd (Iota.Word.ofNat 0xF0) (Iota.Word.ofNat 0x3C) = Iota.Word.ofNat 0x30 ∧ Iota.Word.bitOr (Iota.Word.ofNat 0xF0) (Iota.Word.ofNat 0x3C) = Iota.Word.ofNat 0xFC ∧ Iota.Word.bitXor (Iota.Word.ofNat 0xF0) (Iota.Word.ofNat 0x3C) = Iota.Word.ofNat 0xCC
