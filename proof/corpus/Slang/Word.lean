import Mathlib

namespace Slang

/- The selected source has one numeric carrier.  All constructors pass through
   `ofNat`, so literals and arithmetic have the same modulo-2^64 boundary. -/
def wordBits : Nat := 64
def wordModulus : Nat := 2 ^ wordBits
def wordSignBit : Nat := 2 ^ (wordBits - 1)

abbrev Word := Fin wordModulus

namespace Word

def ofNat (value : Nat) : Word :=
  ⟨value % wordModulus, Nat.mod_lt _ (by simp [wordModulus])⟩

def bits (value : Word) : Nat := value.val

def signed (value : Word) : Int :=
  if value.val < wordSignBit then Int.ofNat value.val
  else Int.ofNat value.val - Int.ofNat wordModulus

def add (left right : Word) : Word := ofNat (left.val + right.val)
def sub (left right : Word) : Word := ofNat (left.val + wordModulus - right.val)
def mul (left right : Word) : Word := ofNat (left.val * right.val)

def bitAnd (left right : Word) : Word := ofNat (left.val &&& right.val)
def bitOr (left right : Word) : Word := ofNat (left.val ||| right.val)
def bitXor (left right : Word) : Word := ofNat (left.val ^^^ right.val)

def bool (condition : Bool) : Word := if condition then ofNat 1 else ofNat 0

def eq (left right : Word) : Word := bool (left == right)
def ne (left right : Word) : Word := bool (left != right)
def lt (left right : Word) : Word := bool (signed left < signed right)
def le (left right : Word) : Word := bool (signed left ≤ signed right)
def gt (left right : Word) : Word := bool (signed left > signed right)
def ge (left right : Word) : Word := bool (signed left ≥ signed right)

def nonzero (value : Word) : Prop := value ≠ ofNat 0

def tablePhi : Word := ofNat 2654435769
def spreadMask : Word := ofNat 4611686018427387903

def spreadPosition (key : Word) : Word :=
  if signed key < 0 then sub (ofNat 0) key else key

/- These definitions fix the source order: signed negativity selects the
   wrapping absolute value, multiplication reduces to `Word`, and only then is
   the product masked. -/
def tableSpreadProduct (key : Word) : Word := mul (spreadPosition key) tablePhi
def tableSpread (key : Word) : Word := bitAnd (tableSpreadProduct key) spreadMask

@[simp] theorem bits_lt_modulus (value : Word) : bits value < wordModulus := value.isLt

@[simp] theorem signed_zero : signed (ofNat 0) = 0 := by norm_num [signed, ofNat, wordSignBit]

@[simp] theorem signed_allOnes : signed (ofNat (wordModulus - 1)) = -1 := by
  norm_num [signed, ofNat, wordSignBit, wordModulus, wordBits]

theorem tableSpread_multiplies_before_mask (key : Word) :
    tableSpread key = bitAnd (mul (spreadPosition key) tablePhi) spreadMask := rfl

theorem tableSpread_is_word_bounded (position : Word) :
    bits (tableSpread position) < wordModulus := (tableSpread position).isLt

theorem wrapping_add_boundary : add (ofNat (wordModulus - 1)) (ofNat 1) = ofNat 0 := by
  norm_num [add, ofNat, wordModulus, wordBits]

theorem wrapping_mul_boundary : mul (ofNat wordSignBit) (ofNat 2) = ofNat 0 := by
  norm_num [mul, ofNat, wordSignBit, wordModulus, wordBits]

theorem signed_comparison_fixture : lt (ofNat (wordModulus - 1)) (ofNat 0) = ofNat 1 := by
  norm_num [lt, bool, signed, ofNat, wordSignBit, wordModulus, wordBits]

theorem bitwise_fixture :
    bitAnd (ofNat 0xF0) (ofNat 0x3C) = ofNat 0x30 ∧
    bitOr (ofNat 0xF0) (ofNat 0x3C) = ofNat 0xFC ∧
    bitXor (ofNat 0xF0) (ofNat 0x3C) = ofNat 0xCC := by
  have hAnd : (240 &&& 60) = 48 := by decide
  have hOr : (240 ||| 60) = 252 := by decide
  have hXor : (240 ^^^ 60) = 204 := by decide
  norm_num [bitAnd, bitOr, bitXor, ofNat, wordModulus, wordBits, hAnd, hOr, hXor]

end Word
end Slang
