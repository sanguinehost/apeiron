/-
  APEIRON-CRYPTO-SPIKE-be — four-byte big-endian decoding bridge.

  `sha256.compress_loop0.body` (`proof/ApeironWord.lean:15243-15264`, from
  `src/sha256.rs:34-40`) writes, at `i < 16`, the word
  `core.num.U32.from_be_bytes (Array.make 4#usize [b0, b1, b2, b3])`.

  This file proves that word is the canonical big-endian regrouping
  `b0·2²⁴ + b1·2¹⁶ + b2·2⁸ + b3`, for arbitrary extracted `Std.U8` bytes, and
  gives an asymmetric byte witness that separates it from the little-endian
  reading and from the all-zero reading.

  The load-bearing step is `or_shiftLeft_eq_add`: the regrouping is a `BitVec`
  `|||` tree, and turning each `|||` into `+` needs disjoint bit positions, not
  bounds; it is proved from the quotient/remainder characterisation of `Nat.or`
  (`Nat.or_div_two_pow`, `Nat.or_mod_two_pow`).

  No `sorry`, `admit`, `axiom`, `native_decide`, `maxHeartbeats` or
  `maxRecDepth` override.

  STATUS: kernel-verified; run 3 of `Sha256BE.md` exits 0 with every `#print axioms`
  row inside {propext, Classical.choice, Quot.sound}, and the LE-order mutation
  control reddens. Evidence, hashes and cones: `Sha256BE.md`.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import ApeironWord

set_option autoImplicit false
set_option linter.unusedVariables false
set_option warningAsError false

namespace Apeiron.Sha256BE

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Sha2 (Word)

/-- One canonical byte. -/
abbrev Byte := BitVec 8

/-! ## Disjoint bit positions -/

/-- If `x < 2^m` then `x` has no set bit at or above position `m`, while
    `y <<< m` has none below it, so the two occupy disjoint positions and their
    bitwise or is their sum. -/
theorem or_shiftLeft_eq_add {x y m : Nat} (hx : x < 2 ^ m) :
    x ||| (y <<< m) = x + (y <<< m) := by
  have h1 : (x ||| (y <<< m)) / 2 ^ m = y := by
    rw [Nat.or_div_two_pow, Nat.div_eq_of_lt hx, Nat.zero_or, Nat.shiftLeft_eq]
    exact Nat.mul_div_cancel y (Nat.two_pow_pos m)
  have h2 : (x ||| (y <<< m)) % 2 ^ m = x := by
    rw [Nat.or_mod_two_pow, Nat.mod_eq_of_lt hx, Nat.shiftLeft_eq, Nat.mul_mod_left,
      Nat.or_zero]
  calc x ||| (y <<< m)
      = (x ||| (y <<< m)) / 2 ^ m * 2 ^ m + (x ||| (y <<< m)) % 2 ^ m :=
        (Nat.div_add_mod' _ _).symm
    _ = y * 2 ^ m + x := by rw [h1, h2]
    _ = x + (y <<< m) := by rw [Nat.shiftLeft_eq]; omega

/-! ## The little-endian byte fold -/

/-- The little-endian value of a byte list: first byte least significant. -/
def leVal : List Byte → Nat
  | [] => 0
  | b :: l => b.toNat + 256 * leVal l

/-- `leVal` is below the width of its list, so the reader never truncates. -/
theorem leVal_lt (l : List Byte) : leVal l < 2 ^ (8 * l.length) := by
  induction l with
  | nil => decide
  | cons b l ih =>
    change b.toNat + 256 * leVal l < 2 ^ (8 * (b :: l).length)
    have hb : b.toNat < 256 := b.isLt
    have h8 : (2 : Nat) ^ 8 = 256 := by decide
    rw [List.length_cons, Nat.mul_add, Nat.mul_one, Nat.pow_add, h8]
    omega

/-- `BitVec.fromLEBytes` is `leVal` at the reader's own width. -/
theorem fromLEBytes_toNat (l : List Byte) : (BitVec.fromLEBytes l).toNat = leVal l := by
  induction l with
  | nil => decide
  | cons b l ih =>
    change (BitVec.setWidth (8 * (b :: l).length) b |||
        (BitVec.setWidth (8 * (b :: l).length) (BitVec.fromLEBytes l) <<< 8)).toNat
      = b.toNat + 256 * leVal l
    have hb : b.toNat < 256 := b.isLt
    have hlt : leVal l < 2 ^ (8 * l.length) := leVal_lt l
    have hw₁ : (8 : Nat) ≤ 8 * (b :: l).length := by
      rw [List.length_cons, Nat.mul_add, Nat.mul_one]; omega
    have hw₂ : 8 * l.length ≤ 8 * (b :: l).length := by
      rw [List.length_cons, Nat.mul_add, Nat.mul_one]; omega
    have h8 : (2 : Nat) ^ 8 = 256 := by decide
    have hshift : leVal l <<< 8 < 2 ^ (8 * (b :: l).length) := by
      rw [Nat.shiftLeft_eq, List.length_cons, Nat.mul_add, Nat.mul_one, Nat.pow_add, h8]
      omega
    rw [BitVec.toNat_or, BitVec.toNat_shiftLeft,
      BitVec.toNat_setWidth_of_le (b := b) hw₁,
      BitVec.toNat_setWidth_of_le (b := BitVec.fromLEBytes l) hw₂,
      ih, Nat.mod_eq_of_lt hshift,
      or_shiftLeft_eq_add (x := b.toNat) (y := leVal l) (m := 8) hb,
      Nat.shiftLeft_eq, h8]
    omega

/-! ## The four-byte bridge -/

/-- The canonical big-endian regrouping of four bytes, in the arithmetic
    spelling of `Crypto.Sha256Certificate.beWord16`. -/
def beWord4 (a b c d : Byte) : Word :=
  BitVec.ofNat 32 (a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat)

theorem beWord4_toNat (a b c d : Byte) :
    (beWord4 a b c d).toNat
      = (a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat) % 2 ^ 32 :=
  BitVec.toNat_ofNat
    (a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat) 32

theorem four_lt (a b c d : Byte) :
    a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat < 2 ^ 32 := by
  have ha : a.toNat < 256 := a.isLt
  have hb : b.toNat < 256 := b.isLt
  have hc : c.toNat < 256 := c.isLt
  have hd : d.toNat < 256 := d.isLt
  have h32 : (2 : Nat) ^ 32 = 4294967296 := by decide
  rw [h32]
  omega

/-- `BitVec.fromBEBytes [a,b,c,d]` is the canonical big-endian regrouping.
    The pinned toolchain has no `BitVec.fromBEBytes_eq`; this is derived. -/
theorem fromBEBytes_toNat_four (a b c d : Byte) :
    (BitVec.fromBEBytes [a, b, c, d]).toNat
      = a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat := by
  show (BitVec.fromLEBytes [d, c, b, a]).toNat
      = a.toNat * 16777216 + b.toNat * 65536 + c.toNat * 256 + d.toNat
  rw [fromLEBytes_toNat]
  simp only [leVal]
  omega

/-- The bridge as a `BitVec` equality. -/
theorem fromBEBytes_four (a b c d : Byte) :
    BitVec.fromBEBytes [a, b, c, d] = beWord4 a b c d := by
  apply BitVec.eq_of_toNat_eq
  rw [fromBEBytes_toNat_four, beWord4_toNat]
  exact (Nat.mod_eq_of_lt (four_lt a b c d)).symm

/-- The same, under the width cast the extraction's model attaches. -/
theorem fromBEBytes_four_cast (a b c d : Byte)
    (h : 8 * ([a, b, c, d] : List Byte).length = 32) :
    (BitVec.fromBEBytes [a, b, c, d]).cast h = beWord4 a b c d := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_cast, fromBEBytes_toNat_four, beWord4_toNat]
  exact (Nat.mod_eq_of_lt (four_lt a b c d)).symm

/-- The extracted four-byte big-endian load of `sha256.compress_loop0.body` is
    the canonical big-endian regrouping.  `core.num.U32.from_be_bytes` is, in
    the pinned Aeneas model, `BitVec.fromBEBytes` of the byte vector
    (`CoreConvertNum.lean:605-606`). -/
theorem from_be_bytes_make4 (a b c d : Std.U8) :
    (core.num.U32.from_be_bytes (Array.make 4#usize [a, b, c, d])).bv
      = beWord4 a.bv b.bv c.bv d.bv := by
  simp only [core.num.U32.from_be_bytes, UScalar.bv_mk_apply]
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_cast, Array.make_val]
  rw [show List.map U8.bv [a, b, c, d] = [a.bv, b.bv, c.bv, d.bv] from rfl]
  rw [fromBEBytes_toNat_four, beWord4_toNat]
  exact (Nat.mod_eq_of_lt (four_lt a.bv b.bv c.bv d.bv)).symm

/-! ## Link to the committed certificate's spelling -/

/-- `beWord16` (`Crypto/Sha256Certificate.lean:484`) is `beWord4` at bytes
    `4i … 4i+3`. -/
theorem beWord16_eq_beWord4 (blk : Array Std.U8 64#usize) (i : Fin 16) :
    apeiron_word.beWord16 blk i
      = beWord4 (apeiron_word.b64 blk (4 * i.val))
          (apeiron_word.b64 blk (4 * i.val + 1))
          (apeiron_word.b64 blk (4 * i.val + 2))
          (apeiron_word.b64 blk (4 * i.val + 3)) := rfl

/-! ## Asymmetric byte witness

The first byte is most significant: `[1,0,0,0]` reads `0x01000000`, the
little-endian reading of the same bytes is `1`, and the big-endian reading is
neither zero (the all-zero reading) nor the little-endian value. -/

theorem from_be_bytes_asym :
    (core.num.U32.from_be_bytes (Array.make 4#usize [1#u8, 0#u8, 0#u8, 0#u8])).bv
        = 0x01000000 ∧
      (BitVec.fromLEBytes [1, 0, 0, 0]).toNat = 1 ∧
      (0x01000000 : BitVec 32) ≠ 0 ∧
      (0x01000000 : BitVec 32) ≠ 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [from_be_bytes_make4]; decide
  · rw [fromLEBytes_toNat]; decide
  · decide
  · decide

/-! ## Kernel checks: axiom cones -/

#print axioms Apeiron.Sha256BE.or_shiftLeft_eq_add
#print axioms Apeiron.Sha256BE.leVal_lt
#print axioms Apeiron.Sha256BE.fromLEBytes_toNat
#print axioms Apeiron.Sha256BE.four_lt
#print axioms Apeiron.Sha256BE.fromBEBytes_toNat_four
#print axioms Apeiron.Sha256BE.beWord4_toNat
#print axioms Apeiron.Sha256BE.fromBEBytes_four
#print axioms Apeiron.Sha256BE.fromBEBytes_four_cast
#print axioms Apeiron.Sha256BE.from_be_bytes_make4
#print axioms Apeiron.Sha256BE.beWord16_eq_beWord4
#print axioms Apeiron.Sha256BE.from_be_bytes_asym

end Apeiron.Sha256BE
