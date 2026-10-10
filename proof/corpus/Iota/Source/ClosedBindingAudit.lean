import Iota.Source.ClosedBinding
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.ClosedBindingAudit

/-- A local fixture whose proof term is `rfl`, so the empty axiom cone is positive. The runner's
mutation control replaces this proof with `Classical.choice`, making it the red/restored-green
target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

#audit_empty Iota.Source.ClosedBinding : Iota.Source.SourceBundle → Iota.Decode.CertifiedDecoded → Prop
#audit_empty Iota.Source.ClosedBinding.intro : ∀ {bundle : Iota.Source.SourceBundle} {value : Iota.Decode.CertifiedDecoded} (entryBytes cellPath cellBytes stdImportPath stdBytes : Iota.Source.SourceBytes) (entryDecoded : Iota.Decode.CertifiedDecoded) (stdTokens : List Iota.Decode.Token) (stdConsumed : Nat) (coord : Iota.ConstantDecl), bundle.fileBytes? bundle.entry = some entryBytes → Iota.Source.ParsesClosure bundle entryDecoded.val.program → Iota.Source.importPaths entryBytes = [cellPath] → bundle.fileBytes? cellPath = some cellBytes → Iota.Source.importPaths cellBytes = [stdImportPath] → bundle.fileBytes? stdImportPath = some stdBytes → Iota.Source.ImportedCoordDerivation stdBytes stdTokens stdConsumed coord → coord.value = .literal 8796093022208 → value = Iota.Decode.appendImportedCoord entryDecoded coord → Iota.Source.ClosedBinding bundle value
#audit_empty Iota.Source.ClosedBinding.sound : ∀ {bundle : Iota.Source.SourceBundle} {value : Iota.Decode.CertifiedDecoded}, Iota.Source.ClosedBinding bundle value → ∃ (entryBytes cellPath cellBytes stdImportPath stdBytes : Iota.Source.SourceBytes) (entryDecoded : Iota.Decode.CertifiedDecoded) (stdTokens : List Iota.Decode.Token) (stdConsumed : Nat) (coord : Iota.ConstantDecl), bundle.fileBytes? bundle.entry = some entryBytes ∧ Iota.Source.ParsesClosure bundle entryDecoded.val.program ∧ Iota.Source.importPaths entryBytes = [cellPath] ∧ bundle.fileBytes? cellPath = some cellBytes ∧ Iota.Source.importPaths cellBytes = [stdImportPath] ∧ bundle.fileBytes? stdImportPath = some stdBytes ∧ Iota.Source.ImportedCoordDerivation stdBytes stdTokens stdConsumed coord ∧ coord.value = .literal 8796093022208 ∧ value = Iota.Decode.appendImportedCoord entryDecoded coord
#audit_empty Iota.Source.ClosedBinding.coordDecodes : ∀ {bundle : Iota.Source.SourceBundle} {value : Iota.Decode.CertifiedDecoded}, Iota.Source.ClosedBinding bundle value → ∃ (stdImportPath stdBytes : Iota.Source.SourceBytes) (coord : Iota.ConstantDecl), bundle.fileBytes? stdImportPath = some stdBytes ∧ Iota.Decode.decodeImportedCoordCeiling (Iota.Source.encodeBytes stdBytes) = .ok coord ∧ coord.value = .literal 8796093022208
#audit_empty Iota.Source.ClosedBinding.missing_lookup_not_closed : ∀ {bundle : Iota.Source.SourceBundle} {value : Iota.Decode.CertifiedDecoded} {entryBytes path : Iota.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → path ∈ Iota.Source.importPaths entryBytes → bundle.fileBytes? path = none → ¬Iota.Source.ClosedBinding bundle value
#audit_empty Iota.Source.ClosedBinding.missing_cell_import_not_closed : ∀ {bundle : Iota.Source.SourceBundle} {value : Iota.Decode.CertifiedDecoded} {entryBytes cellPath cellBytes path : Iota.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → Iota.Source.importPaths entryBytes = [cellPath] → bundle.fileBytes? cellPath = some cellBytes → path ∈ Iota.Source.importPaths cellBytes → bundle.fileBytes? path = none → ¬Iota.Source.ClosedBinding bundle value
#audit_empty Iota.Source.ClosedBindingAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.Source.ClosedBindingAudit
