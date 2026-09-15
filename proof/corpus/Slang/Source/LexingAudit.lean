import Slang.Audit.Empty
import Slang.Source.Lexing
import Slang.Derivation.Lexing

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 2048

namespace Slang.Source.LexingAudit

open Slang Slang.Decode Slang.Source

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

/-- UTF-8 for U+00E9 occupies two source bytes. -/
def multibyteSource : SourceBytes := [sourceByte 0xc3, sourceByte 0xa9]

/-- The multibyte fixture has one token spanning both UTF-8 bytes. -/
def multibyteTokens : List Token :=
  [{ kind := .other 'é', span := ⟨0, 2⟩ }]

/-- A checked derivation of the tokenizer's result for the U+00E9 fixture. -/
theorem multibyte_steps : Slang.Derivation.Lexing.Run
    (Slang.Derivation.Lexing.Request.tokenize (encodeBytes multibyteSource))
    (Except.ok (multibyteTokens, 2)) := by
  unfold Slang.Derivation.Lexing.Run Slang.Derivation.Lexing.body Slang.Derivation.Lexing.lexAsk
  dsimp only []
  rw [Slang.Source.ByteRep.encodeBytes_data]
  exact
    Slang.Derivation.Derives.ask
      (Slang.Derivation.Derives.done (Except.ok ['é']))
      (Slang.Derivation.Derives.ask
        (Slang.Derivation.Derives.ask
          (Slang.Derivation.Derives.done (Except.ok (multibyteTokens, 2)))
          (Slang.Derivation.Derives.done (Except.ok (multibyteTokens, 2))))
        (Slang.Derivation.Derives.done (Except.ok (multibyteTokens, 2))))

/-- The literal fixture carries a checked byte-indexed derivation. -/
theorem multibyte_derivation : LexingDerivation multibyteSource multibyteTokens 2 := by
  refine ⟨multibyte_steps, ?_, rfl⟩
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

#audit_empty Slang.Decode.utf8ByteLength : Char → Nat
#audit_empty Slang.Decode.utf8ByteLengthList : List Char → Nat
#audit_empty Slang.Decode.tokenize : ByteArray → Slang.Decode.Result (List Slang.Decode.Token × Nat)
#audit_empty Slang.Source.ByteIndexedSpans : Nat → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.LexingDerivation : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Prop
#audit_empty Slang.Source.LexingDerivation.mk : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.tokenize (Slang.Source.encodeBytes source)) (Except.ok (tokens, consumed)) → Slang.Source.ByteIndexedSpans source.length tokens → consumed = source.length → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.LexingDerivation.steps : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.tokenize (Slang.Source.encodeBytes source)) (Except.ok (tokens, consumed))
#audit_empty Slang.Source.LexingDerivation.intro : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Decode.tokenize (Slang.Source.encodeBytes source) = .ok (tokens, consumed) → Slang.Source.ByteIndexedSpans source.length tokens → consumed = source.length → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.LexingDerivation.byteIndexed : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → Slang.Source.ByteIndexedSpans source.length tokens
#audit_empty Slang.Source.LexingDerivation.fullConsumption : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → consumed = source.length
#audit_empty Slang.Source.LexingDerivation.sound : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → Slang.Decode.tokenize (Slang.Source.encodeBytes source) = .ok (tokens, consumed)
#audit_empty Slang.Source.LexingDerivation.kinds : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → (Slang.Decode.tokenize (Slang.Source.encodeBytes source)).map (fun result => result.1.map (·.kind)) = .ok (tokens.map (·.kind))
#audit_empty Slang.Source.LexingDerivation.endpoints : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → (tokens.all fun token => token.span.start < token.span.stop && token.span.stop ≤ source.length) = true
#audit_empty Slang.Source.LexingDerivation.consumes : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.LexingDerivation source tokens consumed → consumed = source.length
#audit_empty Slang.Source.LexingDerivation.consumes_rep : ∀ {source : Slang.Source.SourceBytes} {packed : ByteArray} {tokens : List Slang.Decode.Token} {consumed : Nat}, Slang.Source.ByteRep source packed → Slang.Source.LexingDerivation source tokens consumed → consumed = packed.size
#audit_empty Slang.Source.LexingAudit.multibyteSource : Slang.Source.SourceBytes
#audit_empty Slang.Source.LexingAudit.multibyteTokens : List Slang.Decode.Token
#audit_empty Slang.Source.LexingAudit.multibyte_derivation : Slang.Source.LexingDerivation Slang.Source.LexingAudit.multibyteSource Slang.Source.LexingAudit.multibyteTokens 2
#audit_empty Slang.Source.LexingAudit.multibyte_advances_two_bytes : Slang.Decode.tokenize (Slang.Source.encodeBytes Slang.Source.LexingAudit.multibyteSource) = .ok (Slang.Source.LexingAudit.multibyteTokens, 2)
#audit_empty Slang.Source.LexingAudit.multibyte_endpoint_not_char_count : Slang.Source.LexingAudit.multibyteTokens.head?.map (·.span.stop) ≠ some 1
#audit_empty Slang.Source.LexingAudit.unknown_reachable_rejected : (match Slang.Decode.parseExprFuel 4 0 Slang.Source.LexingAudit.multibyteTokens with | .error _ => true | .ok _ => false) = true
#audit_empty Slang.Source.LexingAudit.unknown_not_trivia : (Slang.Decode.tokenize (Slang.Source.encodeBytes Slang.Source.LexingAudit.multibyteSource)).map (fun result => result.1.map (·.kind)) = .ok [.other 'é']

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.Source.LexingAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.LexingAudit
