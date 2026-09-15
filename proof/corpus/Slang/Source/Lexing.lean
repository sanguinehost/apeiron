import Slang.Decode
import Slang.Source.Bytes
import Slang.Derivation.Lexing

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

open Slang Slang.Decode

/-- Token spans are nonempty, bounded byte intervals in source order. -/
def ByteIndexedSpans (size : Nat) (tokens : List Token) : Prop :=
  (tokens.all fun token => token.span.start < token.span.stop && token.span.stop ≤ size) = true ∧
  tokens.Pairwise fun left right => left.span.stop ≤ right.span.start

/-- A checked lexical result over the canonical source-byte view. -/
structure LexingDerivation (source : SourceBytes) (tokens : List Token) (consumed : Nat) : Prop where
  steps : Slang.Derivation.Lexing.Run
    (Slang.Derivation.Lexing.Request.tokenize (encodeBytes source)) (.ok (tokens, consumed))
  byteIndexed : ByteIndexedSpans source.length tokens
  fullConsumption : consumed = source.length

namespace LexingDerivation

/-- The tokenizer's output equation yields a checked derivation. -/
def intro {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    (stream : Decode.tokenize (encodeBytes source) = .ok (tokens, consumed))
    (byteIndexed : ByteIndexedSpans source.length tokens)
    (fullConsumption : consumed = source.length) : LexingDerivation source tokens consumed :=
  ⟨stream ▸ Slang.Derivation.Lexing.tokenize_run (encodeBytes source), byteIndexed,
    fullConsumption⟩

/-- A derivation returns exactly the repaired decoder token stream. -/
theorem sound {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    (derivation : LexingDerivation source tokens consumed) :
    Decode.tokenize (encodeBytes source) = .ok (tokens, consumed) :=
  Slang.Derivation.Lexing.Run.sound derivation.steps

/-- A derivation exposes the token kinds without changing their order. -/
theorem kinds {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    (derivation : LexingDerivation source tokens consumed) :
    (Decode.tokenize (encodeBytes source)).map (fun result => result.1.map (·.kind)) =
      .ok (tokens.map (·.kind)) := by
  rw [derivation.sound]
  rfl

/-- Every derived token endpoint is bounded by the source byte length. -/
theorem endpoints {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    (derivation : LexingDerivation source tokens consumed) :
    (tokens.all fun token => token.span.start < token.span.stop &&
      token.span.stop ≤ source.length) = true :=
  derivation.byteIndexed.1

/-- Derived tokenization consumes every source byte. -/
theorem consumes {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    (derivation : LexingDerivation source tokens consumed) : consumed = source.length :=
  derivation.fullConsumption

/-- Byte representation transfers full consumption to the packed size. -/
theorem consumes_rep {source : SourceBytes} {packed : ByteArray} {tokens : List Token}
    {consumed : Nat} (representation : ByteRep source packed)
    (derivation : LexingDerivation source tokens consumed) : consumed = packed.size := by
  rw [derivation.consumes, representation.size_eq_length]

end LexingDerivation
end Slang.Source
