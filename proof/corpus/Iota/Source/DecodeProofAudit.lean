import Iota.Source.DecodeProof
import Iota.Source.ClosureAudit
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.DecodeProofAudit

open Iota Iota.Decode Iota.Source Iota.Source.ClosureAudit

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

#audit_empty Iota.Source.ImportedCoordDerivation : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Iota.ConstantDecl → Prop
#audit_empty Iota.Source.ImportedCoordDerivation.mk : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {coord : Iota.ConstantDecl}, Iota.Source.LexingDerivation source tokens consumed → Iota.Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] [] = .ok ([coord], []) → Iota.Source.ImportedCoordDerivation source tokens consumed coord
#audit_empty Iota.Source.ImportedCoordDerivation.lexing : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {coord : Iota.ConstantDecl}, Iota.Source.ImportedCoordDerivation source tokens consumed coord → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.ImportedCoordDerivation.scanned : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {coord : Iota.ConstantDecl}, Iota.Source.ImportedCoordDerivation source tokens consumed coord → Iota.Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] [] = .ok ([coord], [])
#audit_empty Iota.Source.ImportedCoordDerivation.sound : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {coord : Iota.ConstantDecl}, Iota.Source.ImportedCoordDerivation source tokens consumed coord → Iota.Decode.decodeImportedCoordCeiling (Iota.Source.encodeBytes source) = .ok coord
#audit_empty Iota.Decode.appendImportedCoord : Iota.Decode.CertifiedDecoded → Iota.ConstantDecl → Iota.Decode.CertifiedDecoded
#audit_empty Iota.Decode.decodeClosure : ByteArray → ByteArray → Iota.Decode.Result Iota.Decode.CertifiedDecoded
#audit_empty Iota.Source.composeDecoded : Iota.Decode.CertifiedDecoded → Iota.ConstantDecl → Iota.Decode.CertifiedDecoded
#audit_empty Iota.Source.DecodeDerivation : Iota.Source.SourceBytes → Iota.Source.SourceBytes → Iota.Decode.CertifiedDecoded → Prop
#audit_empty Iota.Source.DecodeDerivation.intro : ∀ {table std : Iota.Source.SourceBytes} {value : Iota.Decode.CertifiedDecoded} (decoded : Iota.Decode.CertifiedDecoded) (coord : Iota.ConstantDecl), Iota.Decode.decodeTable (Iota.Source.encodeBytes table) = .ok decoded → Iota.Decode.decodeImportedCoordCeiling (Iota.Source.encodeBytes std) = .ok coord → coord.value = .literal 8796093022208 → value = Iota.Source.composeDecoded decoded coord → Iota.Source.DecodeDerivation table std value
#audit_empty Iota.Source.decodeDerivation_of_grammar : ∀ {table std : Iota.Source.SourceBytes} {tableTokens stdTokens : List Iota.Decode.Token} {tableConsumed stdConsumed : Nat} {decoded value : Iota.Decode.CertifiedDecoded} {coord : Iota.ConstantDecl}, Iota.Source.GrammarDerivation table tableTokens tableConsumed decoded → Iota.Source.ImportedCoordDerivation std stdTokens stdConsumed coord → coord.value = .literal 8796093022208 → value = Iota.Source.composeDecoded decoded coord → Iota.Source.DecodeDerivation table std value
#audit_empty Iota.Source.decodeDerivation_sound : ∀ {table std : Iota.Source.SourceBytes} {value : Iota.Decode.CertifiedDecoded}, Iota.Source.DecodeDerivation table std value → Iota.Decode.decodeClosure (Iota.Source.encodeBytes table) (Iota.Source.encodeBytes std) = .ok value
#audit_empty Iota.Source.ParsesClosure.decodeDerivation : ∀ {bundle : Iota.Source.SourceBundle} {program : Iota.Program}, Iota.Source.ParsesClosure bundle program → {stdBytes : Iota.Source.SourceBytes} → bundle.fileBytes? Iota.Source.stdPath = some stdBytes → {coord : Iota.ConstantDecl} → Iota.Decode.decodeImportedCoordCeiling (Iota.Source.encodeBytes stdBytes) = .ok coord → coord.value = .literal 8796093022208 → ∃ entryBytes stdBytesOut coordOut value, bundle.fileBytes? bundle.entry = some entryBytes ∧ bundle.fileBytes? Iota.Source.stdPath = some stdBytesOut ∧ coordOut.value = .literal 8796093022208 ∧ Iota.Source.DecodeDerivation entryBytes stdBytesOut value ∧ value.val.program = { program with constants := program.constants ++ [coordOut] }
#audit_empty Iota.Source.ParsesClosure.decodeClosure_ok : ∀ {bundle : Iota.Source.SourceBundle} {program : Iota.Program}, Iota.Source.ParsesClosure bundle program → {stdBytes : Iota.Source.SourceBytes} → bundle.fileBytes? Iota.Source.stdPath = some stdBytes → {coord : Iota.ConstantDecl} → Iota.Decode.decodeImportedCoordCeiling (Iota.Source.encodeBytes stdBytes) = .ok coord → coord.value = .literal 8796093022208 → ∃ entryBytes stdBytesOut value, bundle.fileBytes? bundle.entry = some entryBytes ∧ bundle.fileBytes? Iota.Source.stdPath = some stdBytesOut ∧ Iota.Decode.decodeClosure (Iota.Source.encodeBytes entryBytes) (Iota.Source.encodeBytes stdBytesOut) = .ok value
#audit_empty Iota.Source.DecodeProofAudit.sound_application : ∀ {table std : Iota.Source.SourceBytes} {value : Iota.Decode.CertifiedDecoded}, Iota.Source.DecodeDerivation table std value → Iota.Decode.decodeClosure (Iota.Source.encodeBytes table) (Iota.Source.encodeBytes std) = .ok value

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Iota.Source.DecodeProofAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.Source.DecodeProofAudit
