import Iota.Generated.TableDerivation
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.SourceAudit

open Iota Iota.Decode Iota.Source

def ExpectedSelectedSyntax (program : Program) : Bool :=
  program.constants.map (·.name) == Decode.selectedConstantNames &&
    program.functions.map (·.name) == Decode.selectedFunctionNames &&
    (program.constants.find? fun declaration => declaration.name == "TABLE_PHI").map (·.value) ==
      some (.literal 2654435769)

def decodedSourceExact : Bool :=
  match Decode.decodeClosure
      (encodeBytes Iota.Generated.TableDerivation.tableSource)
      (encodeBytes Iota.Generated.TableDerivation.stdSource) with
  | .ok value => ExpectedSelectedSyntax value.val.program
  | .error _ => false

-- Compiled whole-source tests; these are not registered proof acceptance.
#guard Iota.Generated.TableDerivation.decodeAccepts
  Iota.Generated.TableDerivation.tableSource
#guard Iota.Source.selectedSourceExact Iota.Generated.TableDerivation.tableSource
#guard Iota.Source.importPaths Iota.Generated.TableDerivation.tableSource ==
  [Iota.Generated.TableDerivation.cellPath]
#guard Iota.Source.importPaths Iota.Generated.TableDerivation.cellSource ==
  [Iota.Generated.TableDerivation.toSource Iota.Generated.Table.stdPath.toUTF8]
#guard (Iota.Generated.TableDerivation.entryBundle.fileBytes?
  Iota.Generated.TableDerivation.cellPath).isSome
#guard (Iota.Generated.TableDerivation.entryBundle.fileBytes?
  Iota.Generated.TableDerivation.stdPath).isSome
#guard decodedSourceExact

/-- The conditional source-binding specification retained independently of the concrete witness. -/
theorem source_binding_of_closed_binding {value : Decode.CertifiedDecoded}
    (derivation : Source.DecodeDerivation
      Iota.Generated.TableDerivation.tableSource
      Iota.Generated.TableDerivation.stdSource value)
    (selectedSyntax : ExpectedSelectedSyntax value.val.program = true)
    (binding : Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value) :
    Decode.decodeClosure
        (Source.encodeBytes Iota.Generated.TableDerivation.tableSource)
        (Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = .ok value ∧
      value.val.certificate.Valid ∧
      ExpectedSelectedSyntax value.val.program = true ∧
      Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by
  exact ⟨Source.decodeDerivation_sound derivation, value.property, selectedSyntax, binding⟩

#audit_empty Iota.SourceAudit.ExpectedSelectedSyntax : Iota.Program → Bool
#audit_empty Iota.SourceAudit.decodedSourceExact : Bool
#audit_empty Iota.SourceAudit.source_binding_of_closed_binding : ∀ {value : Iota.Decode.CertifiedDecoded}, Iota.Source.DecodeDerivation Iota.Generated.TableDerivation.tableSource Iota.Generated.TableDerivation.stdSource value → Iota.SourceAudit.ExpectedSelectedSyntax value.val.program = true → Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value → Iota.Decode.decodeClosure (Iota.Source.encodeBytes Iota.Generated.TableDerivation.tableSource) (Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = .ok value ∧ value.val.certificate.Valid ∧ Iota.SourceAudit.ExpectedSelectedSyntax value.val.program = true ∧ Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Iota.SourceAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.SourceAudit
