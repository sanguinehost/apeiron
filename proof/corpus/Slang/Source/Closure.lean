import Slang.Source.Grammar

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

open Slang Slang.Decode

structure SourceFile where
  path : SourceBytes
  bytes : SourceBytes
  deriving BEq

structure SourceBundle where
  entry : SourceBytes
  files : List SourceFile

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

def stdPath : SourceBytes := [sourceByte 115, sourceByte 116, sourceByte 100]

def SourceBundle.fileBytes? (bundle : SourceBundle) (path : SourceBytes) : Option SourceBytes :=
  (bundle.files.find? fun file => file.path == path).map (·.bytes)

private def reverseAux : SourceBytes → SourceBytes → SourceBytes
  | [], reversed => reversed
  | byte :: rest, reversed => reverseAux rest (byte :: reversed)

private def reverse (bytes : SourceBytes) : SourceBytes := reverseAux bytes []

private def untilQuote : SourceBytes → Option SourceBytes
  | [] => none
  | byte :: rest =>
      if byte.val == 34 then some []
      else
        match untilQuote rest with
        | none => none
        | some path => some (byte :: path)

private def usePrefix (first second third space quote : SourceByte) : Bool :=
  first.val == 117 && second.val == 115 && third.val == 101 &&
    space.val == 32 && quote.val == 34

private def quotedPath (source : SourceBytes) : Option SourceBytes :=
  match source with
  | [] => none
  | first :: afterFirst =>
      match afterFirst with
      | [] => none
      | second :: afterSecond =>
          match afterSecond with
          | [] => none
          | third :: afterThird =>
              match afterThird with
              | [] => none
              | space :: afterSpace =>
                  match afterSpace with
                  | [] => none
                  | quote :: rest =>
                      if usePrefix first second third space quote then untilQuote rest else none

private def prependPath (path : Option SourceBytes) (rest : List SourceBytes) : List SourceBytes :=
  match path with
  | none => rest
  | some value => value :: rest

private def importPathsGo : SourceBytes → SourceBytes → List SourceBytes
  | [], reversedLine => prependPath (quotedPath (reverse reversedLine)) []
  | byte :: rest, reversedLine =>
      if byte.val == 10 then
        prependPath (quotedPath (reverse reversedLine)) (importPathsGo rest [])
      else importPathsGo rest (byte :: reversedLine)

def importPaths (source : SourceBytes) : List SourceBytes := importPathsGo source []

def scanSourceSelected (source : SourceBytes) :
    Decode.Result (List ConstantDecl × List FunctionDecl) := do
  let (tokens, _) ← Decode.tokenize (encodeBytes source)
  Decode.scanSelected (tokens.length + 1) tokens
    Decode.selectedConstantNames Decode.selectedFunctionNames

def selectedTableConstantNames : List String :=
  Decode.selectedConstantNames.erase "COORD_CEILING"

def selectedSourceExact (source : SourceBytes) : Bool :=
  match scanSourceSelected source with
  | .ok (constants, functions) =>
      constants.map (·.name) == selectedTableConstantNames &&
      functions.map (·.name) == Decode.selectedFunctionNames
  | .error _ => false

def SelectedSourceExact (source : SourceBytes) : Prop := selectedSourceExact source = true

def SelectedProgramExact (program : Program) : Prop :=
  program.constants.map (·.name) = selectedTableConstantNames ∧
  program.functions.map (·.name) = Decode.selectedFunctionNames

inductive ParsesClosure (bundle : SourceBundle) (program : Program) : Prop where
  | intro (entryBytes : SourceBytes)
      (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes)
      (decoded : CertifiedDecoded)
      (entryDecodes : Decode.decodeTable (encodeBytes entryBytes) = .ok decoded)
      (programEq : decoded.val.program = program)
      (selectedSource : SelectedSourceExact entryBytes)
      (selectedProgram : SelectedProgramExact program)
      (importsResolved : ∀ path, path ∈ importPaths entryBytes →
        (bundle.fileBytes? path).isSome = true)
      (stdBytes : SourceBytes)
      (stdLookup : bundle.fileBytes? stdPath = some stdBytes)
      (coord : ConstantDecl)
      (coordDecodes : Decode.decodeImportedCoordCeiling (encodeBytes stdBytes) = .ok coord)
      (coordValue : coord.value = .literal 8796093022208) : ParsesClosure bundle program

namespace ParsesClosure

theorem entry_decodes {bundle : SourceBundle} {program : Program}
    (closure : ParsesClosure bundle program) :
    ∃ entryBytes decoded,
      bundle.fileBytes? bundle.entry = some entryBytes ∧
      Decode.decodeTable (encodeBytes entryBytes) = .ok decoded ∧
      decoded.val.program = program := by
  cases closure with
  | intro entryBytes entryLookup decoded entryDecodes programEq _ _ _ _ _ _ _ _ =>
      exact ⟨entryBytes, decoded, entryLookup, entryDecodes, programEq⟩

theorem coord_bound {bundle : SourceBundle} {program : Program}
    (closure : ParsesClosure bundle program) :
    ∃ stdBytes coord,
      bundle.fileBytes? stdPath = some stdBytes ∧
      Decode.decodeImportedCoordCeiling (encodeBytes stdBytes) = .ok coord ∧
      coord.value = .literal 8796093022208 := by
  cases closure with
  | intro _ _ _ _ _ _ _ _ stdBytes stdLookup coord coordDecodes coordValue =>
      exact ⟨stdBytes, coord, stdLookup, coordDecodes, coordValue⟩

end ParsesClosure

theorem unresolved_import_not_parses {bundle : SourceBundle} {program : Program}
    {entryBytes path : SourceBytes}
    (entry : bundle.fileBytes? bundle.entry = some entryBytes)
    (edge : path ∈ importPaths entryBytes)
    (missing : bundle.fileBytes? path = none) :
    ¬ParsesClosure bundle program := by
  intro closure
  cases closure with
  | intro closureBytes closureLookup _ _ _ _ _ importsResolved _ _ _ _ _ =>
      have lookupEq : some closureBytes = some entryBytes := closureLookup.symm.trans entry
      have bytesEq : closureBytes = entryBytes := Option.some.inj lookupEq
      have resolved := importsResolved path (bytesEq ▸ edge)
      have missingSome : (bundle.fileBytes? path).isSome = false :=
        congrArg Option.isSome missing
      exact Bool.noConfusion (resolved.symm.trans missingSome)

def DuplicateOrSkippedSelected (source : SourceBytes) : Prop :=
  selectedSourceExact source = false

theorem duplicate_or_skipped_selected_not_parses {bundle : SourceBundle} {program : Program}
    {entryBytes : SourceBytes}
    (entry : bundle.fileBytes? bundle.entry = some entryBytes)
    (inexact : DuplicateOrSkippedSelected entryBytes) :
    ¬ParsesClosure bundle program := by
  intro closure
  cases closure with
  | intro closureBytes closureLookup _ _ _ selectedSource _ _ _ _ _ _ _ =>
      have lookupEq : some closureBytes = some entryBytes := closureLookup.symm.trans entry
      have bytesEq : closureBytes = entryBytes := Option.some.inj lookupEq
      have selectedEntry : selectedSourceExact entryBytes = true :=
        congrArg selectedSourceExact bytesEq.symm |>.trans selectedSource
      exact Bool.noConfusion (selectedEntry.symm.trans inexact)

end Slang.Source
