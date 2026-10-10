import Iota.Source.Grammar

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source

open Iota Iota.Decode

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

-- Temporal O(f·p) for f = |bundle.files| and p = |path|: `List.find?` tests `file.path == path`
-- once per file until the first hit, each `BEq (List SourceByte)` walking at most p bytes.
-- Spatial O(1): `find?` returns a subterm and its recursion is tail; `.map` allocates one `Option`.
def SourceBundle.fileBytes? (bundle : SourceBundle) (path : SourceBytes) : Option SourceBytes :=
  (bundle.files.find? fun file => file.path == path).map (·.bytes)

-- Temporal O(n) for n = |bytes|: one `::` step per input byte, closing on the finite list.
-- Spatial O(n) for the n-cell reversed result; the recursion is tail-recursive, so O(1) call stack.
private def reverseAux : SourceBytes → SourceBytes → SourceBytes
  | [], reversed => reversed
  | byte :: rest, reversed => reverseAux rest (byte :: reversed)

-- Temporal O(n) and spatial O(n) as `reverseAux bytes []`: one cons per byte and an n-cell result,
-- with O(1) call stack.
private def reverse (bytes : SourceBytes) : SourceBytes := reverseAux bytes []

-- Temporal O(k) for k bytes scanned to the first `"` (or to the end): one value comparison per
-- byte. Spatial O(k): the returned prefix path holds k cons cells and the recursion binds its
-- result, so k stack frames.
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

-- Temporal O(k) for k = |source|: the five-byte prefix match is O(1), then `untilQuote` scans the
-- remainder once. Spatial O(k) for the returned path and the `untilQuote` stack.
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

-- Temporal O(n) for n = |source|: each byte is consed onto `reversedLine` once, and each completed
-- line is reversed and scanned once, so every byte is charged O(1) across the recursion, which
-- closes on the finite input. Spatial O(n) for the line accumulator, the result paths and the
-- non-tail frames, one per line.
private def importPathsGo : SourceBytes → SourceBytes → List SourceBytes
  | [], reversedLine => prependPath (quotedPath (reverse reversedLine)) []
  | byte :: rest, reversedLine =>
      if byte.val == 10 then
        prependPath (quotedPath (reverse reversedLine)) (importPathsGo rest [])
      else importPathsGo rest (byte :: reversedLine)

-- Temporal O(n) and spatial O(n) as `importPathsGo source []`: one linear pass over the source and
-- the accumulated paths.
def importPaths (source : SourceBytes) : List SourceBytes := importPathsGo source []

-- Temporal O(n) for n = |source|: `encodeBytes` and `Decode.tokenize` each make one linear pass,
-- and `Decode.scanSelected` takes `tokens.length + 1` steps, one per token. Spatial O(n) for the
-- packed bytes, decoded characters, token list and scan stack.
def scanSourceSelected (source : SourceBytes) :
    Decode.Result (List ConstantDecl × List FunctionDecl) := do
  let (tokens, _) ← Decode.tokenize (encodeBytes source)
  Decode.scanSelected (tokens.length + 1) tokens
    Decode.selectedConstantNames Decode.selectedFunctionNames

def selectedTableConstantNames : List String :=
  Decode.selectedConstantNames.erase "COORD_CEILING"

-- Temporal O(n) for n = |source|: `scanSourceSelected` is linear, then the two `map`/`==`
-- comparisons walk the selected constants and functions (at most one per token). Spatial O(n),
-- dominated by `scanSourceSelected`, plus the mapped name lists.
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

inductive ParsesClosure (bundle : SourceBundle) (entryProgram : Program) : Prop where
  | intro (entryBytes : SourceBytes)
      (entryLookup : bundle.fileBytes? bundle.entry = some entryBytes)
      (entryDecoded : CertifiedDecoded)
      (entryDecodes : Decode.decodeTable (encodeBytes entryBytes) = .ok entryDecoded)
      (programEq : entryDecoded.val.program = entryProgram)
      (selectedSource : SelectedSourceExact entryBytes)
      (selectedProgram : SelectedProgramExact entryProgram)
      (importsResolved : ∀ path, path ∈ importPaths entryBytes →
        (bundle.fileBytes? path).isSome = true) : ParsesClosure bundle entryProgram

namespace ParsesClosure

theorem entry_decodes {bundle : SourceBundle} {program : Program}
    (closure : ParsesClosure bundle program) :
    ∃ entryBytes decoded,
      bundle.fileBytes? bundle.entry = some entryBytes ∧
      Decode.decodeTable (encodeBytes entryBytes) = .ok decoded ∧
      decoded.val.program = program := by
  cases closure with
  | intro entryBytes entryLookup decoded entryDecodes programEq _ _ _ =>
      exact ⟨entryBytes, decoded, entryLookup, entryDecodes, programEq⟩

end ParsesClosure

theorem unresolved_import_not_parses {bundle : SourceBundle} {program : Program}
    {entryBytes path : SourceBytes}
    (entry : bundle.fileBytes? bundle.entry = some entryBytes)
    (edge : path ∈ importPaths entryBytes)
    (missing : bundle.fileBytes? path = none) :
    ¬ParsesClosure bundle program := by
  intro closure
  cases closure with
  | intro closureBytes closureLookup _ _ _ _ _ importsResolved =>
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
  | intro closureBytes closureLookup _ _ _ selectedSource _ _ =>
      have lookupEq : some closureBytes = some entryBytes := closureLookup.symm.trans entry
      have bytesEq : closureBytes = entryBytes := Option.some.inj lookupEq
      have selectedEntry : selectedSourceExact entryBytes = true :=
        congrArg selectedSourceExact bytesEq.symm |>.trans selectedSource
      exact Bool.noConfusion (selectedEntry.symm.trans inexact)

end Iota.Source
