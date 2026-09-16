import Slang.Source.DecodeProof
import Slang.Source.ClosureAudit
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.DecodeProofAudit

open Slang Slang.Decode Slang.Source Slang.Source.ClosureAudit

private def completeStd : SourceBytes := asciiBytes "COORD_CEILING=8796093022208\n"

private def corruptStd : SourceBytes := asciiBytes "COORD_CEILING=8796093022207\n"

private def decodeViaDerivation (table std : SourceBytes) : Decode.Result CertifiedDecoded := do
  let (tableTokens, tableConsumed) ← Decode.tokenize (encodeBytes table)
  let decoded ← Decode.decodeTokens (encodeBytes table) tableTokens tableConsumed
  let (stdTokens, _) ← Decode.tokenize (encodeBytes std)
  let (constants, functions) ← Decode.scanSelected (stdTokens.length + 1) stdTokens
    ["COORD_CEILING"] []
  match constants, functions with
  | [coord], [] =>
      if coord.value != .literal 8796093022208 then
        .error "COORD_CEILING has the wrong value"
      else
        .ok (composeDecoded decoded coord)
  | _, _ => .error "COORD_CEILING is missing or duplicated"

private def acceptsDerivationPath (table std : SourceBytes) : Bool :=
  match decodeViaDerivation table std with
  | .ok _ => true
  | .error _ => false

private def derivationPathMatchesDecoder (table std : SourceBytes) : Bool :=
  match decodeViaDerivation table std,
      Decode.decodeClosure (encodeBytes table) (encodeBytes std) with
  | .ok derived, .ok decoded => derived.val == decoded.val
  | _, _ => false

#guard acceptsDerivationPath completeEntry completeStd
#guard derivationPathMatchesDecoder completeEntry completeStd
#guard !acceptsDerivationPath completeEntry corruptStd

theorem sound_application {table std : SourceBytes} {value : CertifiedDecoded}
    (derivation : DecodeDerivation table std value) :
    Decode.decodeClosure (encodeBytes table) (encodeBytes std) = .ok value :=
  decodeDerivation_sound derivation

#audit_empty Slang.Source.ImportedCoordDerivation : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Slang.ConstantDecl → Prop
#audit_empty Slang.Source.ImportedCoordDerivation.mk : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {coord : Slang.ConstantDecl}, Slang.Source.LexingDerivation source tokens consumed → Slang.Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] [] = .ok ([coord], []) → Slang.Source.ImportedCoordDerivation source tokens consumed coord
#audit_empty Slang.Source.ImportedCoordDerivation.lexing : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {coord : Slang.ConstantDecl}, Slang.Source.ImportedCoordDerivation source tokens consumed coord → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.ImportedCoordDerivation.scanned : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {coord : Slang.ConstantDecl}, Slang.Source.ImportedCoordDerivation source tokens consumed coord → Slang.Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] [] = .ok ([coord], [])
#audit_empty Slang.Source.ImportedCoordDerivation.sound : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {coord : Slang.ConstantDecl}, Slang.Source.ImportedCoordDerivation source tokens consumed coord → Slang.Decode.decodeImportedCoordCeiling (Slang.Source.encodeBytes source) = .ok coord
#audit_empty Slang.Decode.appendImportedCoord : Slang.Decode.CertifiedDecoded → Slang.ConstantDecl → Slang.Decode.CertifiedDecoded
#audit_empty Slang.Decode.decodeClosure : ByteArray → ByteArray → Slang.Decode.Result Slang.Decode.CertifiedDecoded
#audit_empty Slang.Source.composeDecoded : Slang.Decode.CertifiedDecoded → Slang.ConstantDecl → Slang.Decode.CertifiedDecoded
#audit_empty Slang.Source.DecodeDerivation : Slang.Source.SourceBytes → Slang.Source.SourceBytes → Slang.Decode.CertifiedDecoded → Prop
#audit_empty Slang.Source.DecodeDerivation.intro : ∀ {table std : Slang.Source.SourceBytes} {value : Slang.Decode.CertifiedDecoded} (decoded : Slang.Decode.CertifiedDecoded) (coord : Slang.ConstantDecl), Slang.Decode.decodeTable (Slang.Source.encodeBytes table) = .ok decoded → Slang.Decode.decodeImportedCoordCeiling (Slang.Source.encodeBytes std) = .ok coord → coord.value = .literal 8796093022208 → value = Slang.Source.composeDecoded decoded coord → Slang.Source.DecodeDerivation table std value
#audit_empty Slang.Source.decodeDerivation_of_grammar : ∀ {table std : Slang.Source.SourceBytes} {tableTokens stdTokens : List Slang.Decode.Token} {tableConsumed stdConsumed : Nat} {decoded value : Slang.Decode.CertifiedDecoded} {coord : Slang.ConstantDecl}, Slang.Source.GrammarDerivation table tableTokens tableConsumed decoded → Slang.Source.ImportedCoordDerivation std stdTokens stdConsumed coord → coord.value = .literal 8796093022208 → value = Slang.Source.composeDecoded decoded coord → Slang.Source.DecodeDerivation table std value
#audit_empty Slang.Source.decodeDerivation_sound : ∀ {table std : Slang.Source.SourceBytes} {value : Slang.Decode.CertifiedDecoded}, Slang.Source.DecodeDerivation table std value → Slang.Decode.decodeClosure (Slang.Source.encodeBytes table) (Slang.Source.encodeBytes std) = .ok value
#audit_empty Slang.Source.ParsesClosure.decodeDerivation : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program}, Slang.Source.ParsesClosure bundle program → {stdBytes : Slang.Source.SourceBytes} → bundle.fileBytes? Slang.Source.stdPath = some stdBytes → {coord : Slang.ConstantDecl} → Slang.Decode.decodeImportedCoordCeiling (Slang.Source.encodeBytes stdBytes) = .ok coord → coord.value = .literal 8796093022208 → ∃ entryBytes stdBytesOut coordOut value, bundle.fileBytes? bundle.entry = some entryBytes ∧ bundle.fileBytes? Slang.Source.stdPath = some stdBytesOut ∧ coordOut.value = .literal 8796093022208 ∧ Slang.Source.DecodeDerivation entryBytes stdBytesOut value ∧ value.val.program = { program with constants := program.constants ++ [coordOut] }
#audit_empty Slang.Source.ParsesClosure.decodeClosure_ok : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program}, Slang.Source.ParsesClosure bundle program → {stdBytes : Slang.Source.SourceBytes} → bundle.fileBytes? Slang.Source.stdPath = some stdBytes → {coord : Slang.ConstantDecl} → Slang.Decode.decodeImportedCoordCeiling (Slang.Source.encodeBytes stdBytes) = .ok coord → coord.value = .literal 8796093022208 → ∃ entryBytes stdBytesOut value, bundle.fileBytes? bundle.entry = some entryBytes ∧ bundle.fileBytes? Slang.Source.stdPath = some stdBytesOut ∧ Slang.Decode.decodeClosure (Slang.Source.encodeBytes entryBytes) (Slang.Source.encodeBytes stdBytesOut) = .ok value
#audit_empty Slang.Source.DecodeProofAudit.sound_application : ∀ {table std : Slang.Source.SourceBytes} {value : Slang.Decode.CertifiedDecoded}, Slang.Source.DecodeDerivation table std value → Slang.Decode.decodeClosure (Slang.Source.encodeBytes table) (Slang.Source.encodeBytes std) = .ok value

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.Source.DecodeProofAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.DecodeProofAudit
