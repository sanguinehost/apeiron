import Slang.Generated.TableDerivation
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.SourceAudit

open Slang Slang.Decode Slang.Source

def ExpectedSelectedSyntax (program : Program) : Bool :=
  program.constants.map (·.name) == Decode.selectedConstantNames &&
    program.functions.map (·.name) == Decode.selectedFunctionNames &&
    (program.constants.find? fun declaration => declaration.name == "TABLE_PHI").map (·.value) ==
      some (.literal 2654435769)

def decodedSourceExact : Bool :=
  match Decode.decodeClosure
      (encodeBytes Slang.Generated.TableDerivation.tableSource)
      (encodeBytes Slang.Generated.TableDerivation.stdSource) with
  | .ok value => ExpectedSelectedSyntax value.val.program
  | .error _ => false

-- Compiled whole-source tests; these are not registered proof acceptance.
#guard Slang.Generated.TableDerivation.decodeAccepts
  Slang.Generated.TableDerivation.tableSource
#guard Slang.Source.selectedSourceExact Slang.Generated.TableDerivation.tableSource
#guard Slang.Source.importPaths Slang.Generated.TableDerivation.tableSource ==
  [Slang.Generated.TableDerivation.cellPath]
#guard Slang.Source.importPaths Slang.Generated.TableDerivation.cellSource ==
  [Slang.Generated.TableDerivation.toSource Slang.Generated.Table.stdPath.toUTF8]
#guard (Slang.Generated.TableDerivation.entryBundle.fileBytes?
  Slang.Generated.TableDerivation.cellPath).isSome
#guard (Slang.Generated.TableDerivation.entryBundle.fileBytes? Slang.Source.stdPath).isSome
#guard decodedSourceExact

/-- A checked producer derivation, syntax check, and closure binding imply the source headline. -/
theorem source_binding {value : Decode.CertifiedDecoded}
    (derivation : Source.DecodeDerivation
      Slang.Generated.TableDerivation.tableSource
      Slang.Generated.TableDerivation.stdSource value)
    (selectedSyntax : ExpectedSelectedSyntax value.val.program = true)
    (closure : Source.ParsesClosure
      Slang.Generated.TableDerivation.entryBundle value.val.program) :
    Decode.decodeClosure
        (Source.encodeBytes Slang.Generated.TableDerivation.tableSource)
        (Source.encodeBytes Slang.Generated.TableDerivation.stdSource) = .ok value ∧
      value.val.certificate.Valid ∧
      ExpectedSelectedSyntax value.val.program = true ∧
      Source.ParsesClosure Slang.Generated.TableDerivation.entryBundle value.val.program := by
  exact ⟨Source.decodeDerivation_sound derivation, value.property, selectedSyntax, closure⟩

#audit_empty Slang.SourceAudit.ExpectedSelectedSyntax : Slang.Program → Bool
#audit_empty Slang.SourceAudit.decodedSourceExact : Bool
#audit_empty Slang.SourceAudit.source_binding : ∀ {value : Slang.Decode.CertifiedDecoded}, Slang.Source.DecodeDerivation Slang.Generated.TableDerivation.tableSource Slang.Generated.TableDerivation.stdSource value → Slang.SourceAudit.ExpectedSelectedSyntax value.val.program = true → Slang.Source.ParsesClosure Slang.Generated.TableDerivation.entryBundle value.val.program → Slang.Decode.decodeClosure (Slang.Source.encodeBytes Slang.Generated.TableDerivation.tableSource) (Slang.Source.encodeBytes Slang.Generated.TableDerivation.stdSource) = .ok value ∧ value.val.certificate.Valid ∧ Slang.SourceAudit.ExpectedSelectedSyntax value.val.program = true ∧ Slang.Source.ParsesClosure Slang.Generated.TableDerivation.entryBundle value.val.program

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.SourceAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.SourceAudit
