import Iota.Source.DecodeProof

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source

open Iota Iota.Decode

/-- The closed source binding.

`ParsesClosure bundle entryProgram` is entry-indexed: it carries only the entry decode and the
entry-source checks, and its selected constant-name list excludes `COORD_CEILING`. This predicate
composes that entry binding with the checked textual import path `table -> cell -> std`, the bundle
lookups for each file on the path, the `ImportedCoordDerivation` for the literal
`8796093022208`, and the append equation tying the composed value to the entry decode and the
imported coordinate. It is defined in its own module so that `DecodeProof` (which imports
`Closure`) can be imported here without a cycle; `ClosedBinding` is the index that the closed
program (nine selected constants, `COORD_CEILING` included) inhabits. -/
inductive ClosedBinding (bundle : SourceBundle) (value : CertifiedDecoded) : Prop where
  | intro (entryBytes cellPath cellBytes stdImportPath stdBytes : SourceBytes)
      (entryDecoded : CertifiedDecoded)
      (stdTokens : List Token) (stdConsumed : Nat) (coord : ConstantDecl)
      (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes)
      (entryBinding : ParsesClosure bundle entryDecoded.val.program)
      (tableImports : importPaths entryBytes = [cellPath])
      (cellLookup : bundle.fileBytes? cellPath = some cellBytes)
      (cellImports : importPaths cellBytes = [stdImportPath])
      (stdLookup : bundle.fileBytes? stdImportPath = some stdBytes)
      (coordDerivation : ImportedCoordDerivation stdBytes stdTokens stdConsumed coord)
      (coordValue : coord.value = .literal 8796093022208)
      (composed : value = Decode.appendImportedCoord entryDecoded coord) :
      ClosedBinding bundle value

namespace ClosedBinding

/-- Every field of a `ClosedBinding` holds. The binding exposes its entry file, the
`table -> cell -> std` textual edges, the bundle lookups for each file on that path, the entry
`ParsesClosure`, the imported-coordinate derivation and the append equation. -/
theorem sound {bundle : SourceBundle} {value : CertifiedDecoded}
    (binding : ClosedBinding bundle value) :
    ∃ (entryBytes cellPath cellBytes stdImportPath stdBytes : SourceBytes)
      (entryDecoded : CertifiedDecoded) (stdTokens : List Token) (stdConsumed : Nat)
      (coord : ConstantDecl),
      bundle.fileBytes? bundle.entry = some entryBytes ∧
      ParsesClosure bundle entryDecoded.val.program ∧
      importPaths entryBytes = [cellPath] ∧
      bundle.fileBytes? cellPath = some cellBytes ∧
      importPaths cellBytes = [stdImportPath] ∧
      bundle.fileBytes? stdImportPath = some stdBytes ∧
      ImportedCoordDerivation stdBytes stdTokens stdConsumed coord ∧
      coord.value = .literal 8796093022208 ∧
      value = Decode.appendImportedCoord entryDecoded coord := by
  cases binding with
  | intro entryBytes cellPath cellBytes stdImportPath stdBytes entryDecoded
      stdTokens stdConsumed coord entryLookup entryBinding tableImports cellLookup
      cellImports stdLookup coordDerivation coordValue composed =>
    exact ⟨entryBytes, cellPath, cellBytes, stdImportPath, stdBytes, entryDecoded,
      stdTokens, stdConsumed, coord, entryLookup, entryBinding, tableImports, cellLookup,
      cellImports, stdLookup, coordDerivation, coordValue, composed⟩

/-- The imported `COORD_CEILING` coordinate of a `ClosedBinding` decodes from its `std` source
bytes and carries the literal value `8796093022208`. -/
theorem coordDecodes {bundle : SourceBundle} {value : CertifiedDecoded}
    (binding : ClosedBinding bundle value) :
    ∃ (stdImportPath stdBytes : SourceBytes) (coord : ConstantDecl),
      bundle.fileBytes? stdImportPath = some stdBytes ∧
      Decode.decodeImportedCoordCeiling (encodeBytes stdBytes) = .ok coord ∧
      coord.value = .literal 8796093022208 := by
  cases binding with
  | intro _ _ _ stdImportPath stdBytes _ _ _ coord _ _ _ _ _ stdLookup coordDerivation
      coordValue _ =>
    exact ⟨stdImportPath, stdBytes, coord, stdLookup, coordDerivation.sound, coordValue⟩

/-- A `table` import edge that the bundle cannot resolve cannot inhabit a `ClosedBinding`: the
entry's textual imports are exactly the single `cell` path, so any member of `importPaths
entryBytes` must be the resolved `cell` file. This is the `ClosedBinding` analogue of
`unresolved_import_not_parses`. -/
theorem missing_lookup_not_closed {bundle : SourceBundle} {value : CertifiedDecoded}
    {entryBytes path : SourceBytes}
    (entry : bundle.fileBytes? bundle.entry = some entryBytes)
    (edge : path ∈ importPaths entryBytes)
    (missing : bundle.fileBytes? path = none) :
    ¬ ClosedBinding bundle value := by
  intro binding
  cases binding with
  | intro entryBytes' cellPath' cellBytes' _ _ _ _ _ _ entryLookup' _ tableImports'
      cellLookup' _ _ _ _ _ =>
    have bytesEq : entryBytes' = entryBytes := Option.some.inj (entryLookup'.symm.trans entry)
    have tableImportsEntry : importPaths entryBytes = [cellPath'] := by
      rw [← bytesEq]; exact tableImports'
    have pathEq : path = cellPath' := by
      have member : path ∈ [cellPath'] := tableImportsEntry ▸ edge
      cases member with
      | head => rfl
      | tail _ impossible => cases impossible
    have lookupSome : bundle.fileBytes? path = some cellBytes' := by
      rw [pathEq]; exact cellLookup'
    have impossible : (some cellBytes' : Option SourceBytes) = none :=
      lookupSome.symm.trans missing
    cases impossible

/-- A `cell -> std` import edge that the bundle cannot resolve cannot inhabit a `ClosedBinding`:
the resolved `cell` file's textual imports are exactly the single `std` path, so any member of
`importPaths cellBytes` must be the resolved `std` file. -/
theorem missing_cell_import_not_closed {bundle : SourceBundle} {value : CertifiedDecoded}
    {entryBytes cellPath cellBytes path : SourceBytes}
    (entry : bundle.fileBytes? bundle.entry = some entryBytes)
    (tableEdge : importPaths entryBytes = [cellPath])
    (cell : bundle.fileBytes? cellPath = some cellBytes)
    (edge : path ∈ importPaths cellBytes)
    (missing : bundle.fileBytes? path = none) :
    ¬ ClosedBinding bundle value := by
  intro binding
  cases binding with
  | intro entryBytes' cellPath' cellBytes' stdImportPath' stdBytes' _ _ _ _
      entryLookup' _ tableImports' cellLookup' cellImports' stdLookup' _ _ _ =>
    have bytesEq : entryBytes' = entryBytes := Option.some.inj (entryLookup'.symm.trans entry)
    have tableImportsEntry : importPaths entryBytes = [cellPath'] := by
      rw [← bytesEq]; exact tableImports'
    have cellPathEq : cellPath' = cellPath := by
      have singletonEq : [cellPath'] = [cellPath] := tableImportsEntry.symm.trans tableEdge
      have headEq : (some cellPath' : Option SourceBytes) = some cellPath :=
        congrArg List.head? singletonEq
      exact Option.some.inj headEq
    have lookupSome : bundle.fileBytes? cellPath = some cellBytes' := by
      rw [← cellPathEq]; exact cellLookup'
    have cellBytesEq : cellBytes' = cellBytes := Option.some.inj (lookupSome.symm.trans cell)
    have edgeSingleton : path ∈ [stdImportPath'] := by
      have member : path ∈ importPaths cellBytes' := by
        rw [cellBytesEq]; exact edge
      rw [cellImports'] at member
      exact member
    have pathEq : path = stdImportPath' := by
      cases edgeSingleton with
      | head => rfl
      | tail _ impossible => cases impossible
    have lookupSomeStd : bundle.fileBytes? path = some stdBytes' := by
      rw [pathEq]; exact stdLookup'
    have impossible : (some stdBytes' : Option SourceBytes) = none :=
      lookupSomeStd.symm.trans missing
    cases impossible

end ClosedBinding

end Iota.Source
