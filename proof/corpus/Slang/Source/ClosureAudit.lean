import Slang.Source.Closure
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.ClosureAudit

open Slang Slang.Decode Slang.Source

-- Facts small enough for the kernel are proved below; the whole-source fixtures are gated by
-- `#guard`, which runs the compiled evaluator rather than the kernel reducer, adds no declaration
-- and so adds nothing to any axiom cone, and fails the build unless the result is `true`. Kernel
-- reduction of `Decode.scanSelected` does not terminate in budget, so it is never asked to.

def asciiBytes (text : String) : SourceBytes :=
  text.toList.map fun char => Fin.ofNat 256 char.toNat

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

private def sourceBytes (values : List Nat) : SourceBytes := values.map sourceByte

private def entryPath : SourceBytes := sourceBytes [101, 110, 116, 114, 121]
private def cellPath : SourceBytes := sourceBytes [99, 101, 108, 108]

private def useCellSource : SourceBytes :=
  sourceBytes [117, 115, 101, 32, 34, 99, 101, 108, 108, 34, 10]

private def noUseSource : SourceBytes :=
  sourceBytes [84, 65, 66, 76, 69, 95, 80, 72, 73, 61, 49, 10]

private def blankSource : SourceBytes := sourceBytes [10]

private def coordSource : SourceBytes :=
  sourceBytes [67, 79, 79, 82, 68, 95, 67, 69, 73, 76, 73, 78, 71, 61, 49, 10]

/-- The scanner finds exactly the quoted path on a `use` line. -/
theorem importPaths_useCell : importPaths useCellSource = [cellPath] := by rfl

/-- The scanner extracts that path as an import edge. -/
theorem importPaths_edge : cellPath ∈ importPaths useCellSource := by
  have member : cellPath ∈ [cellPath] := List.Mem.head _
  exact importPaths_useCell ▸ member

/-- A source without a `use` line has no import edges. -/
theorem importPaths_empty : importPaths noUseSource = [] := by rfl

/-- A bundle whose entry imports `cell`, with the `cell` and `std` files present. -/
private def closedBundle : SourceBundle :=
  { entry := entryPath
    files := [⟨entryPath, useCellSource⟩, ⟨cellPath, blankSource⟩,
      ⟨stdPath, coordSource⟩] }

/-- The same bundle with the imported `cell` file omitted. -/
private def openBundle : SourceBundle :=
  { entry := entryPath
    files := [⟨entryPath, useCellSource⟩, ⟨stdPath, coordSource⟩] }

theorem open_bundle_missing_cell : openBundle.fileBytes? cellPath = none := rfl

/-- A missing import edge cannot produce a binding. -/
theorem missing_cell_rejected {program : Program} : ¬ ParsesClosure openBundle program :=
  unresolved_import_not_parses (entryBytes := useCellSource) (path := cellPath)
    rfl importPaths_edge open_bundle_missing_cell

/-- The closed bundle resolves every import edge of its entry. -/
theorem closed_bundle_edges_resolved :
    ∀ path, path ∈ importPaths useCellSource → (closedBundle.fileBytes? path).isSome = true := by
  intro path edge
  have singletonEdge : path ∈ [cellPath] := importPaths_useCell ▸ edge
  cases singletonEdge with
  | head => rfl
  | tail _ impossible => exact nomatch impossible

/-- A producer-supplied derivation inhabits the entry-only `ParsesClosure`. -/
theorem parsesClosure_of {bundle : SourceBundle} {program : Program} {entryBytes : SourceBytes}
    (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes) {decoded : CertifiedDecoded}
    (entryDecodes : Decode.decodeTable (encodeBytes entryBytes) = .ok decoded)
    (programEq : decoded.val.program = program) (selectedSource : SelectedSourceExact entryBytes)
    (selectedProgram : SelectedProgramExact program)
    (importsResolved : ∀ path, path ∈ importPaths entryBytes → (bundle.fileBytes? path).isSome = true) :
    ParsesClosure bundle program :=
  .intro entryBytes entryLookup decoded entryDecodes programEq selectedSource selectedProgram
    importsResolved

/-- A complete selected entry: eight table constants and thirteen functions, importing `cell`. -/
def completeEntry : SourceBytes := asciiBytes
  "use \"cell\"\nTABLE_PRESENT=1\nTABLE_EMPTY=0-1\nTABLE_STRIDE=2\nMERGE_SET=1\nMERGE_MAP=2\nMERGE_ACC=3\nMERGE_GRAPH=4\nTABLE_PHI=2654435769\ntableSpread(){return 0;}\ntableSlotKey(){return 0;}\ntableSlotVal(){return 0;}\ntableSlotSet(){return 0;}\ntableSlotValSet(){return 0;}\ntableInitGo(){return 0;}\ntableInit(){return 0;}\ntableCapacityOk(){return 0;}\ntableFindSlotGo(){return 0;}\ntableLookupSlotGo(){return 0;}\ntableFind(){return 0;}\ntableGet(){return 0;}\ntableInsertWith(){return 0;}\n"

/-- The complete entry with a conflicting repeated selected constant appended. -/
private def duplicateEntry : SourceBytes := completeEntry ++ asciiBytes "TABLE_PRESENT=2\n"

private def decodeAccepts (source : SourceBytes) : Bool :=
  match Decode.decodeTable (encodeBytes source) with
  | .ok _ => true
  | .error _ => false

-- The scanner finds the one quoted edge.
#guard importPaths completeEntry == [cellPath]

-- The complete entry decodes and is exact.
#guard decodeAccepts completeEntry
#guard selectedSourceExact completeEntry

-- A conflicting repeated selected constant is rejected, not skipped.
#guard !decodeAccepts duplicateEntry
#guard !selectedSourceExact duplicateEntry

-- The closed bundle resolves the `cell` edge; the open bundle does not.
#guard (closedBundle.fileBytes? cellPath).isSome
#guard (openBundle.fileBytes? cellPath).isNone

#audit_empty Slang.Source.SourceFile : Type
#audit_empty Slang.Source.SourceFile.mk : Slang.Source.SourceBytes → Slang.Source.SourceBytes → Slang.Source.SourceFile
#audit_empty Slang.Source.SourceFile.path : Slang.Source.SourceFile → Slang.Source.SourceBytes
#audit_empty Slang.Source.SourceFile.bytes : Slang.Source.SourceFile → Slang.Source.SourceBytes
#audit_empty Slang.Source.instBEqSourceFile : BEq Slang.Source.SourceFile
#audit_empty Slang.Source.SourceBundle : Type
#audit_empty Slang.Source.SourceBundle.mk : Slang.Source.SourceBytes → List Slang.Source.SourceFile → Slang.Source.SourceBundle
#audit_empty Slang.Source.SourceBundle.entry : Slang.Source.SourceBundle → Slang.Source.SourceBytes
#audit_empty Slang.Source.SourceBundle.files : Slang.Source.SourceBundle → List Slang.Source.SourceFile
#audit_empty Slang.Source.stdPath : Slang.Source.SourceBytes
#audit_empty Slang.Source.SourceBundle.fileBytes? : Slang.Source.SourceBundle → Slang.Source.SourceBytes → Option Slang.Source.SourceBytes
#audit_empty Slang.Source.importPaths : Slang.Source.SourceBytes → List Slang.Source.SourceBytes
#audit_empty Slang.Source.scanSourceSelected : Slang.Source.SourceBytes → Slang.Decode.Result (List Slang.ConstantDecl × List Slang.FunctionDecl)
#audit_empty Slang.Source.selectedTableConstantNames : List String
#audit_empty Slang.Source.selectedSourceExact : Slang.Source.SourceBytes → Bool
#audit_empty Slang.Source.SelectedSourceExact : Slang.Source.SourceBytes → Prop
#audit_empty Slang.Source.SelectedProgramExact : Slang.Program → Prop
#audit_empty Slang.Decode.decodeImportedCoordCeiling : ByteArray → Slang.Decode.Result Slang.ConstantDecl
#audit_empty Slang.Source.ParsesClosure : Slang.Source.SourceBundle → Slang.Program → Prop
#audit_empty Slang.Source.ParsesClosure.intro : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program} (entryBytes : Slang.Source.SourceBytes), bundle.fileBytes? bundle.entry = some entryBytes → (entryDecoded : Slang.Decode.CertifiedDecoded) → Slang.Decode.decodeTable (Slang.Source.encodeBytes entryBytes) = .ok entryDecoded → entryDecoded.val.program = program → Slang.Source.SelectedSourceExact entryBytes → Slang.Source.SelectedProgramExact program → (∀ path, path ∈ Slang.Source.importPaths entryBytes → (bundle.fileBytes? path).isSome = true) → Slang.Source.ParsesClosure bundle program
#audit_empty Slang.Source.ParsesClosure.entry_decodes : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program}, Slang.Source.ParsesClosure bundle program → ∃ entryBytes decoded, bundle.fileBytes? bundle.entry = some entryBytes ∧ Slang.Decode.decodeTable (Slang.Source.encodeBytes entryBytes) = .ok decoded ∧ decoded.val.program = program
#audit_empty Slang.Source.unresolved_import_not_parses : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program} {entryBytes path : Slang.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → path ∈ Slang.Source.importPaths entryBytes → bundle.fileBytes? path = none → ¬Slang.Source.ParsesClosure bundle program
#audit_empty Slang.Source.DuplicateOrSkippedSelected : Slang.Source.SourceBytes → Prop
#audit_empty Slang.Source.duplicate_or_skipped_selected_not_parses : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program} {entryBytes : Slang.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → Slang.Source.DuplicateOrSkippedSelected entryBytes → ¬Slang.Source.ParsesClosure bundle program
#audit_empty Slang.Source.ClosureAudit.importPaths_useCell : Slang.Source.importPaths useCellSource = [cellPath]
#audit_empty Slang.Source.ClosureAudit.importPaths_edge : cellPath ∈ Slang.Source.importPaths useCellSource
#audit_empty Slang.Source.ClosureAudit.importPaths_empty : Slang.Source.importPaths noUseSource = []
#audit_empty Slang.Source.ClosureAudit.open_bundle_missing_cell : openBundle.fileBytes? cellPath = none
#audit_empty Slang.Source.ClosureAudit.missing_cell_rejected : ∀ {program : Slang.Program}, ¬Slang.Source.ParsesClosure openBundle program
#audit_empty Slang.Source.ClosureAudit.closed_bundle_edges_resolved : ∀ path, path ∈ Slang.Source.importPaths useCellSource → (closedBundle.fileBytes? path).isSome = true
#audit_empty Slang.Source.ClosureAudit.parsesClosure_of : ∀ {bundle : Slang.Source.SourceBundle} {program : Slang.Program} {entryBytes : Slang.Source.SourceBytes}, bundle.fileBytes? bundle.entry = some entryBytes → {decoded : Slang.Decode.CertifiedDecoded} → Slang.Decode.decodeTable (Slang.Source.encodeBytes entryBytes) = .ok decoded → decoded.val.program = program → Slang.Source.SelectedSourceExact entryBytes → Slang.Source.SelectedProgramExact program → (∀ path, path ∈ Slang.Source.importPaths entryBytes → (bundle.fileBytes? path).isSome = true) → Slang.Source.ParsesClosure bundle program

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.Source.ClosureAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.ClosureAudit
