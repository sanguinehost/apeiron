import Iota.Audit.Empty
import Iota.Source.Lexing

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.LexingAudit

open Iota Iota.Decode Iota.Source

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

/-- UTF-8 for U+00E9 occupies two source bytes. -/
def multibyteSource : SourceBytes := [sourceByte 0xc3, sourceByte 0xa9]

/-- The multibyte fixture has one token spanning both UTF-8 bytes. -/
def multibyteTokens : List Token :=
  [{ kind := .other 'é', span := ⟨0, 2⟩ }]

/-- A checked derivation of the tokenizer's result for the U+00E9 fixture. -/
theorem multibyte_steps :
    Tokenizer.TokenizeDerivation multibyteSource multibyteTokens 2 := by
  apply Tokenizer.TokenizeDerivation.intro
  · exact Tokenizer.Utf8Scan.next
      (Tokenizer.Utf8Head.two _ _ [] (by decide) (by decide) rfl)
      Tokenizer.Utf8Scan.done
  · exact Tokenizer.CharsScan.nextToken
      (Tokenizer.LexHead.other 'é' [] 0 rfl (by decide) (by decide) rfl rfl rfl rfl rfl
        rfl rfl rfl rfl rfl rfl rfl rfl)
      (Tokenizer.CharsScan.done (Tokenizer.ReverseTokens.result _))

/-- The literal fixture carries a checked byte-indexed derivation. -/
theorem multibyte_derivation : LexingDerivation multibyteSource multibyteTokens 2 := by
  refine LexingDerivation.intro multibyte_steps ?_ rfl
  constructor
  · rfl
  · exact List.Pairwise.cons (fun _ membership => nomatch membership) List.Pairwise.nil

/-- The repaired decoder advances U+00E9 by two bytes. -/
theorem multibyte_advances_two_bytes :
    Decode.tokenize (encodeBytes multibyteSource) = .ok (multibyteTokens, 2) :=
  multibyte_derivation.sound

/-- The byte endpoint differs from the decoded character count. -/
theorem multibyte_endpoint_not_char_count :
    multibyteTokens.head?.map (·.span.stop) ≠ some 1 := by decide

/-- A reachable unknown token enters the rejecting expression branch. -/
theorem unknown_reachable_rejected :
    (match Decode.parseExprFuel 4 0 multibyteTokens with
      | .error _ => true
      | .ok _ => false) = true := rfl

/-- The multibyte token is present rather than discarded as trivia. -/
theorem unknown_not_trivia :
    (Decode.tokenize (encodeBytes multibyteSource)).map
      (fun result => result.1.map (·.kind)) = .ok [.other 'é'] :=
  multibyte_derivation.kinds

#audit_empty Iota.Decode.utf8ByteLength : Char → Nat
#audit_empty Iota.Decode.utf8ByteLengthList : List Char → Nat
#audit_empty Iota.Decode.tokenize : ByteArray → Iota.Decode.Result (List Iota.Decode.Token × Nat)
#audit_empty Iota.Source.ByteIndexedSpans : Nat → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.LexingDerivation : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Prop
#audit_empty Iota.Source.LexingDerivation.mk : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.Tokenizer.TokenizeDerivation source tokens consumed → Iota.Source.ByteIndexedSpans source.length tokens → consumed = source.length → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.LexingDerivation.steps : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → Iota.Source.Tokenizer.TokenizeDerivation source tokens consumed
#audit_empty Iota.Source.LexingDerivation.intro : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.Tokenizer.TokenizeDerivation source tokens consumed → Iota.Source.ByteIndexedSpans source.length tokens → consumed = source.length → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.LexingDerivation.byteIndexed : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → Iota.Source.ByteIndexedSpans source.length tokens
#audit_empty Iota.Source.LexingDerivation.fullConsumption : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → consumed = source.length
#audit_empty Iota.Source.LexingDerivation.sound : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → Iota.Decode.tokenize (Iota.Source.encodeBytes source) = .ok (tokens, consumed)
#audit_empty Iota.Source.LexingDerivation.kinds : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → (Iota.Decode.tokenize (Iota.Source.encodeBytes source)).map (fun result => result.1.map (·.kind)) = .ok (tokens.map (·.kind))
#audit_empty Iota.Source.LexingDerivation.endpoints : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → (tokens.all fun token => token.span.start < token.span.stop && token.span.stop ≤ source.length) = true
#audit_empty Iota.Source.LexingDerivation.consumes : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.LexingDerivation source tokens consumed → consumed = source.length
#audit_empty Iota.Source.LexingDerivation.consumes_rep : ∀ {source : Iota.Source.SourceBytes} {packed : ByteArray} {tokens : List Iota.Decode.Token} {consumed : Nat}, Iota.Source.ByteRep source packed → Iota.Source.LexingDerivation source tokens consumed → consumed = packed.size
#audit_empty Iota.Source.LexingAudit.multibyteSource : Iota.Source.SourceBytes
#audit_empty Iota.Source.LexingAudit.multibyteTokens : List Iota.Decode.Token
#audit_empty Iota.Source.LexingAudit.multibyte_derivation : Iota.Source.LexingDerivation Iota.Source.LexingAudit.multibyteSource Iota.Source.LexingAudit.multibyteTokens 2
#audit_empty Iota.Source.LexingAudit.multibyte_advances_two_bytes : Iota.Decode.tokenize (Iota.Source.encodeBytes Iota.Source.LexingAudit.multibyteSource) = .ok (Iota.Source.LexingAudit.multibyteTokens, 2)
#audit_empty Iota.Source.LexingAudit.multibyte_endpoint_not_char_count : Iota.Source.LexingAudit.multibyteTokens.head?.map (·.span.stop) ≠ some 1
#audit_empty Iota.Source.LexingAudit.unknown_reachable_rejected : (match Iota.Decode.parseExprFuel 4 0 Iota.Source.LexingAudit.multibyteTokens with | .error _ => true | .ok _ => false) = true
#audit_empty Iota.Source.LexingAudit.unknown_not_trivia : (Iota.Decode.tokenize (Iota.Source.encodeBytes Iota.Source.LexingAudit.multibyteSource)).map (fun result => result.1.map (·.kind)) = .ok [.other 'é']

#print axioms Iota.Source.LexingDerivation
#print axioms Iota.Source.LexingDerivation.mk
#print axioms Iota.Source.LexingDerivation.steps
#print axioms Iota.Source.LexingDerivation.intro
#print axioms Iota.Source.LexingDerivation.sound

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Iota.Source.LexingAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.Source.LexingAudit
