import Init.Data.Nat.Div.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Slang

def wordBits : Nat := 64
def wordModulus : Nat := 2 ^ wordBits
def wordSignBit : Nat := 2 ^ (wordBits - 1)

theorem wordModulus_pos : 0 < wordModulus := by
  exact Nat.pow_pos (by exact Nat.zero_lt_succ 1)

abbrev Word := Fin wordModulus

namespace Word

def ofNat (value : Nat) : Word :=
  ⟨value % wordModulus, Nat.mod_lt value wordModulus_pos⟩

def bits (value : Word) : Nat := value.val

def signed (value : Word) : Int :=
  if value.val < wordSignBit then Int.ofNat value.val
  else Int.ofNat value.val - Int.ofNat wordModulus

def add (left right : Word) : Word := ofNat (left.val + right.val)
def sub (left right : Word) : Word := ofNat (left.val + wordModulus - right.val)
def mul (left right : Word) : Word := ofNat (left.val * right.val)

-- Temporal O(fuel): one recursive step per fuel decrement, each doing O(1) division/mod and one
-- `operation` call, so fuel bounds the steps. Spatial O(fuel) stack: the recursive result is bound
-- to `higher` before the conditional, so the recursion is not tail-recursive. Operations counted:
-- fuel decrements / bit extractions (`left % 2`, `left / 2`), one `operation` call per bit. The
-- callers `bitAnd`/`bitOr`/`bitXor` pass `wordBits = 64`, so each is O(1) with a 64-step constant.
def bitwise (operation : Bool → Bool → Bool) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, left, right =>
      let higher := bitwise operation fuel (left / 2) (right / 2)
      let low := operation (left % 2 == 1) (right % 2 == 1)
      if low then higher + higher + 1 else higher + higher

def bitAnd (left right : Word) : Word :=
  ofNat (bitwise Bool.and wordBits left.val right.val)

def bitOr (left right : Word) : Word :=
  ofNat (bitwise Bool.or wordBits left.val right.val)

def bitXor (left right : Word) : Word :=
  ofNat (bitwise Bool.xor wordBits left.val right.val)

def decidableEq : DecidableEq Word := fun left right =>
  match Nat.decEq left.val right.val with
  | isTrue equal => isTrue (Fin.eq_of_val_eq equal)
  | isFalse unequal => isFalse (fun wordsEqual => unequal (congrArg Fin.val wordsEqual))

instance instDecidableEq : DecidableEq Word := decidableEq

def beq (left right : Word) : Bool := Nat.beq left.val right.val

instance instBEq : BEq Word := ⟨beq⟩

instance instInhabited : Inhabited Word := ⟨ofNat 0⟩

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

def tableSpreadProduct (key : Word) : Word := mul (spreadPosition key) tablePhi
def tableSpread (key : Word) : Word := bitAnd (tableSpreadProduct key) spreadMask

@[simp] theorem ofNat_val (value : Nat) :
    (ofNat value).val = value % wordModulus := rfl

@[simp] theorem bits_lt_modulus (value : Word) : bits value < wordModulus := value.isLt

theorem signed_view (value : Word) :
    signed value =
      if value.val < wordSignBit then Int.ofNat value.val
      else Int.ofNat value.val - Int.ofNat wordModulus := rfl

theorem wrapping_add_agreement (left right : Word) :
    (add left right).val = (left.val + right.val) % wordModulus := rfl

theorem wrapping_sub_agreement (left right : Word) :
    (sub left right).val = (left.val + wordModulus - right.val) % wordModulus := rfl

theorem wrapping_mul_agreement (left right : Word) :
    (mul left right).val = (left.val * right.val) % wordModulus := rfl

theorem bitAnd_agreement (left right : Word) :
    (bitAnd left right).val =
      bitwise Bool.and wordBits left.val right.val % wordModulus := rfl

theorem bitOr_agreement (left right : Word) :
    (bitOr left right).val =
      bitwise Bool.or wordBits left.val right.val % wordModulus := rfl

theorem bitXor_agreement (left right : Word) :
    (bitXor left right).val =
      bitwise Bool.xor wordBits left.val right.val % wordModulus := rfl

@[simp] theorem signed_zero : signed (ofNat 0) = 0 := rfl

@[simp] theorem signed_allOnes : signed (ofNat (wordModulus - 1)) = -1 := rfl

theorem tableSpread_multiplies_before_mask (key : Word) :
    tableSpread key = bitAnd (mul (spreadPosition key) tablePhi) spreadMask := rfl

theorem tableSpread_is_word_bounded (position : Word) :
    bits (tableSpread position) < wordModulus := bits_lt_modulus (tableSpread position)

theorem wrapping_add_boundary : add (ofNat (wordModulus - 1)) (ofNat 1) = ofNat 0 := rfl

theorem wrapping_mul_boundary : mul (ofNat wordSignBit) (ofNat 2) = ofNat 0 := rfl

theorem signed_comparison_fixture : lt (ofNat (wordModulus - 1)) (ofNat 0) = ofNat 1 := rfl

theorem bitwise_fixture :
    bitAnd (ofNat 0xF0) (ofNat 0x3C) = ofNat 0x30 ∧
    bitOr (ofNat 0xF0) (ofNat 0x3C) = ofNat 0xFC ∧
    bitXor (ofNat 0xF0) (ofNat 0x3C) = ofNat 0xCC := ⟨rfl, rfl, rfl⟩

end Word
end Slang
