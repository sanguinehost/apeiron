import Slang.Source.ClosedBinding
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.ClosedBindingAudit

/-- A local fixture whose proof term is `rfl`, so the empty axiom cone is positive. The runner's
mutation control replaces this proof with `Classical.choice`, making it the red/restored-green
target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

#audit_empty Slang.Source.ClosedBinding : Slang.Source.SourceBundle → Slang.Decode.CertifiedDecoded → Prop
#audit_empty Slang.Source.ClosedBinding.intro : ∀ {bundle : Slang.Source.SourceBundle} {value : Slang.Decode.CertifiedDecoded} (entryBytes cellPath cellBytes stdImportPath stdBytes : Slang.Source.SourceBytes) (entryDecoded : Slang.Decode.CertifiedDecoded) (stdTokens : List Slang.Decode.Token) (stdConsumed : Nat) (coord : Slang.ConstantDecl), bundle.fileBytes? bundle.entry = some entryBytes → Slang.Source.ParsesClosure bundle entryDecoded.val.program → Slang.Source.importPaths entryBytes = [cellPath] → bundle.fileBytes? cellPath = some cellBytes → Slang.Source.importPaths cellBytes = [stdImportPath] → bundle.fileBytes? stdImportPath = some stdBytes → Slang.Source.ImportedCoordDerivation stdBytes stdTokens stdConsumed coord → coord.value = .literal 8796093022208 → value = Slang.Decode.appendImportedCoord entryDecoded coord → Slang.Source.ClosedBinding bundle value
#audit_empty Slang.Source.ClosedBinding.sound : ∀ {bundle : Slang.Source.SourceBundle} {value : Slang.Decode.CertifiedDecoded}, Slang.Source.ClosedBinding bundle value → ∃ (entryBytes cellPath cellBytes stdImportPath stdBytes : Slang.Source.SourceBytes) (entryDecoded : Slang.Decode.CertifiedDecoded) (stdTokens : List Slang.Decode.Token) (stdConsumed : Nat) (coord : Slang.ConstantDecl), bundle.fileBytes? bundle.entry = some entryBytes ∧ Slang.Source.ParsesClosure bundle entryDecoded.val.program ∧ Slang.Source.importPaths entryBytes = [cellPath] ∧ bundle.fileBytes? cellPath = some cellBytes ∧ Slang.Source.importPaths cellBytes = [stdImportPath] ∧ bundle.fileBytes? stdImportPath = some stdBytes ∧ Slang.Source.ImportedCoordDerivation stdBytes stdTokens stdConsumed coord ∧ coord.value = .literal 8796093022208 ∧ value = Slang.Decode.appendImportedCoord entryDecoded coord
#audit_empty Slang.Source.ClosedBinding.coordDecodes : ∀ {bundle : Slang.Source.SourceBundle} {value : Slang.Decode.CertifiedDecoded}, Slang.Source.ClosedBinding bundle value → ∃ (stdImportPath stdBytes : Slang.Source.SourceBytes) (coord : Slang.ConstantDecl), bundle.fileBytes? stdImportPath = some stdBytes ∧ Slang.Decode.decodeImportedCoordCeiling (Slang.Source.encodeBytes stdBytes) = .ok coord ∧ coord.value = .literal 8796093022208
#audit_empty Slang.Source.ClosedBinding.missing_lookup_not_closed : ∀ {bundle : Slang.Source.SourceBundle} {value : Slang.Decode.CertifiedDecoded} {entryBytes path : Slang.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → path ∈ Slang.Source.importPaths entryBytes → bundle.fileBytes? path = none → ¬Slang.Source.ClosedBinding bundle value
#audit_empty Slang.Source.ClosedBinding.missing_cell_import_not_closed : ∀ {bundle : Slang.Source.SourceBundle} {value : Slang.Decode.CertifiedDecoded} {entryBytes cellPath cellBytes path : Slang.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → Slang.Source.importPaths entryBytes = [cellPath] → bundle.fileBytes? cellPath = some cellBytes → path ∈ Slang.Source.importPaths cellBytes → bundle.fileBytes? path = none → ¬Slang.Source.ClosedBinding bundle value
#audit_empty Slang.Source.ClosedBindingAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.ClosedBindingAudit
