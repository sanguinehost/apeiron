import Slang.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Tokenizer

/-- A maximal prefix accepted by `p`, followed by the untouched suffix. -/
inductive TakeWhileScan (p : Char → Bool) : List Char → List Char → List Char → Prop where
  | nil : TakeWhileScan p [] [] []
  | stop {c : Char} {cs : List Char} (rejected : p c = false) :
      TakeWhileScan p (c :: cs) [] (c :: cs)
  | next {c : Char} {cs front rest : List Char} (accepted : p c = true)
      (tail : TakeWhileScan p cs front rest) : TakeWhileScan p (c :: cs) (c :: front) rest

namespace TakeWhileScan

/-- A structural maximal-prefix scan entails the executable helper's result. -/
theorem sound {p : Char → Bool} {input front rest : List Char}
    (scan : TakeWhileScan p input front rest) :
    Decode.takeWhile p input = (front, rest) := by
  induction scan with
  | nil => rfl
  | stop rejected =>
      unfold Decode.takeWhile
      rw [rejected, if_neg Bool.false_ne_true]
  | next accepted _ tailSound =>
      unfold Decode.takeWhile
      rw [accepted, if_pos rfl]
      dsimp only
      rw [tailSound]

end TakeWhileScan

/-- A scan through a comment body, stopping before its newline or at end of input. -/
inductive SkipCommentScan : List Char → Nat → List Char → Nat → Prop where
  | done {pos : Nat} : SkipCommentScan [] pos [] pos
  | newline {cs : List Char} {pos : Nat} : SkipCommentScan ('\n' :: cs) pos ('\n' :: cs) pos
  | next {c : Char} {cs rest : List Char} {pos next : Nat} (notNewline : c ≠ '\n')
      (tail : SkipCommentScan cs (pos + Decode.utf8ByteLength c) rest next) :
      SkipCommentScan (c :: cs) pos rest next

namespace SkipCommentScan

/-- A structural comment scan entails the executable helper's result. -/
theorem sound {input rest : List Char} {pos next : Nat}
    (scan : SkipCommentScan input pos rest next) :
    Decode.skipComment input pos = (rest, next) := by
  induction scan with
  | done => rfl
  | newline => rfl
  | next notNewline _ tailSound =>
      rw [Decode.skipComment.eq_3]
      exact tailSound
      exact notNewline

end SkipCommentScan

/-- A left fold of decimal digits from an explicit accumulator. -/
inductive DigitsFold : Nat → List Char → Nat → Prop where
  | done {acc : Nat} : DigitsFold acc [] acc
  | next {acc value : Nat} {digit : Char} {digits : List Char}
      (tail : DigitsFold (acc * 10 + (digit.toNat - '0'.toNat)) digits value) :
      DigitsFold acc (digit :: digits) value

namespace DigitsFold

/-- A structural decimal fold entails the corresponding `List.foldl`. -/
theorem sound {acc value : Nat} {digits : List Char} (fold : DigitsFold acc digits value) :
    digits.foldl (fun result digit => result * 10 + (digit.toNat - '0'.toNat)) acc = value := by
  induction fold with
  | done => rfl
  | next _ tailSound => exact tailSound

end DigitsFold

/-- A left fold of UTF-8 character widths from an explicit accumulator. -/
inductive Utf8LengthFold : Nat → List Char → Nat → Prop where
  | done {acc : Nat} : Utf8LengthFold acc [] acc
  | next {acc size : Nat} {c : Char} {chars : List Char}
      (tail : Utf8LengthFold (acc + Decode.utf8ByteLength c) chars size) :
      Utf8LengthFold acc (c :: chars) size

namespace Utf8LengthFold

/-- A structural UTF-8 length fold entails the corresponding `List.foldl`. -/
theorem sound {acc size : Nat} {chars : List Char} (fold : Utf8LengthFold acc chars size) :
    chars.foldl (fun result c => result + Decode.utf8ByteLength c) acc = size := by
  induction fold with
  | done => rfl
  | next _ tailSound => exact tailSound

end Utf8LengthFold

end Slang.Source.Tokenizer
