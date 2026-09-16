import Slang.Source.Bytes
import Slang.Source.Tokenizer.Utf8
import Slang.Source.Tokenizer.Scan

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Tokenizer

open Slang.Source

/-- Mapping source bytes to `UInt8` preserves length. -/
theorem sourceBytes_map_length (source : SourceBytes) :
    (source.map UInt8.ofFin).length = source.length := by
  induction source with
  | nil => rfl
  | cons _ tail ih =>
      change Nat.succ (tail.map UInt8.ofFin).length = Nat.succ tail.length
      exact congrArg Nat.succ ih

/-- Encoding preserves the source length in the unpacked byte view. -/
theorem encodeBytes_data_length (source : SourceBytes) :
    (encodeBytes source).data.toList.length = source.length := by
  rw [ByteRep.encodeBytes_data]
  exact sourceBytes_map_length source

/-- Successful structural UTF-8 and character scans for the executable tokenizer. -/
inductive TokenizeDerivation (source : SourceBytes) (tokens : List Decode.Token)
    (consumed : Nat) : Prop where
  | intro {chars : List Char}
      (utf8 : Utf8Scan (source.length + 1) (source.map UInt8.ofFin) chars)
      (charsScan : CharsScan (chars.length + 1) chars 0 [] tokens consumed) :
      TokenizeDerivation source tokens consumed

namespace TokenizeDerivation

/-- A structural tokenizer derivation entails the executable tokenizer result. -/
theorem sound {source : SourceBytes} {tokens : List Decode.Token} {consumed : Nat}
    (derivation : TokenizeDerivation source tokens consumed) :
    Decode.tokenize (encodeBytes source) = .ok (tokens, consumed) := by
  cases derivation with
  | intro utf8 charsScan =>
      unfold Decode.tokenize Decode.decodeUtf8Chars
      rw [ByteRep.encodeBytes_data, sourceBytes_map_length, utf8.sound]
      exact charsScan.sound

end TokenizeDerivation

end Slang.Source.Tokenizer
