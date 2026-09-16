import Slang.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Tokenizer

open Slang.Decode

/-- One successful UTF-8 scalar decode, with every branch guard exposed as a premise. -/
inductive Utf8Head : List UInt8 → Char → List UInt8 → Prop where
  | one (first : UInt8) (rest : List UInt8)
      (ascii : first.toNat < 0x80) :
      Utf8Head (first :: rest) (Char.ofNat first.toNat) rest
  | two (first second : UInt8) (rest : List UInt8)
      (notAscii : ¬ first.toNat < 0x80)
      (belowThree : first.toNat < 0xe0)
      (valid : (first.toNat < 0xc2 ||
        !Decode.isContinuationByte second.toNat) = false) :
      Utf8Head (first :: second :: rest)
        (Char.ofNat ((first.toNat - 0xc0) * 0x40 + (second.toNat - 0x80))) rest
  | three (first second third : UInt8) (rest : List UInt8)
      (notAscii : ¬ first.toNat < 0x80)
      (notTwo : ¬ first.toNat < 0xe0)
      (belowFour : first.toNat < 0xf0)
      (valid : (!Decode.isContinuationByte second.toNat ||
        !Decode.isContinuationByte third.toNat ||
        ((first.toNat - 0xe0) * 0x1000 + (second.toNat - 0x80) * 0x40 +
          (third.toNat - 0x80) < 0x800) ||
        (0xd800 ≤ (first.toNat - 0xe0) * 0x1000 +
            (second.toNat - 0x80) * 0x40 + (third.toNat - 0x80) &&
          (first.toNat - 0xe0) * 0x1000 + (second.toNat - 0x80) * 0x40 +
            (third.toNat - 0x80) < 0xe000)) = false) :
      Utf8Head (first :: second :: third :: rest)
        (Char.ofNat ((first.toNat - 0xe0) * 0x1000 +
          (second.toNat - 0x80) * 0x40 + (third.toNat - 0x80))) rest
  | four (first second third fourth : UInt8) (rest : List UInt8)
      (notAscii : ¬ first.toNat < 0x80)
      (notTwo : ¬ first.toNat < 0xe0)
      (notThree : ¬ first.toNat < 0xf0)
      (belowFive : first.toNat < 0xf5)
      (valid : (!Decode.isContinuationByte second.toNat ||
        !Decode.isContinuationByte third.toNat ||
        !Decode.isContinuationByte fourth.toNat ||
        ((first.toNat - 0xf0) * 0x40000 + (second.toNat - 0x80) * 0x1000 +
          (third.toNat - 0x80) * 0x40 + (fourth.toNat - 0x80) < 0x10000) ||
        (0x110000 ≤ (first.toNat - 0xf0) * 0x40000 +
          (second.toNat - 0x80) * 0x1000 + (third.toNat - 0x80) * 0x40 +
          (fourth.toNat - 0x80))) = false) :
      Utf8Head (first :: second :: third :: fourth :: rest)
        (Char.ofNat ((first.toNat - 0xf0) * 0x40000 +
          (second.toNat - 0x80) * 0x1000 + (third.toNat - 0x80) * 0x40 +
          (fourth.toNat - 0x80))) rest

namespace Utf8Head

/-- A structural head and a successful recursive tail compose to one decoder step. -/
theorem sound {bytes : List UInt8} {c : Char} {rest : List UInt8}
    (head : Utf8Head bytes c rest) {n : Nat} {chars : List Char}
    (tail : Decode.decodeUtf8CharsFuel n rest = .ok chars) :
    Decode.decodeUtf8CharsFuel (n + 1) bytes = .ok (c :: chars) := by
  cases head with
  | one first rest ascii =>
      unfold Decode.decodeUtf8CharsFuel
      rw [if_pos ascii, tail]
      rfl
  | two first second rest notAscii belowThree valid =>
      unfold Decode.decodeUtf8CharsFuel
      rw [if_neg notAscii, if_pos belowThree]
      dsimp only
      rw [valid, tail]
      rfl
  | three first second third rest notAscii notTwo belowFour valid =>
      unfold Decode.decodeUtf8CharsFuel
      rw [if_neg notAscii, if_neg notTwo, if_pos belowFour]
      dsimp only
      rw [valid, tail]
      rfl
  | four first second third fourth rest notAscii notTwo notThree belowFive valid =>
      unfold Decode.decodeUtf8CharsFuel
      rw [if_neg notAscii, if_neg notTwo, if_neg notThree, if_pos belowFive]
      dsimp only
      rw [valid, tail]
      rfl

end Utf8Head

/-- A successful structural scan mirrors the decoder's one-fuel-unit-per-character recursion. -/
inductive Utf8Scan : Nat → List UInt8 → List Char → Prop where
  | done {n : Nat} : Utf8Scan (n + 1) [] []
  | next {n : Nat} {bytes rest : List UInt8} {c : Char} {chars : List Char}
      (head : Utf8Head bytes c rest) (tail : Utf8Scan n rest chars) :
      Utf8Scan (n + 1) bytes (c :: chars)

namespace Utf8Scan

/-- A structural UTF-8 scan entails the result of the executable decoder recursion. -/
theorem sound {n : Nat} {bytes : List UInt8} {chars : List Char}
    (scan : Utf8Scan n bytes chars) :
    Decode.decodeUtf8CharsFuel n bytes = .ok chars := by
  induction scan with
  | done => rfl
  | next head _ tailSound => exact head.sound tailSound

end Utf8Scan

end Slang.Source.Tokenizer
