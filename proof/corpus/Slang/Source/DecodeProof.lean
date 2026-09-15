import Slang.Source.Closure

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

open Slang Slang.Decode

structure ImportedCoordDerivation (source : SourceBytes) (tokens : List Token) (consumed : Nat)
    (coord : ConstantDecl) : Prop where
  lexing : LexingDerivation source tokens consumed
  scanned : Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] [] =
    .ok ([coord], [])

namespace ImportedCoordDerivation

theorem sound {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {coord : ConstantDecl}
    (derivation : ImportedCoordDerivation source tokens consumed coord) :
    Decode.decodeImportedCoordCeiling (encodeBytes source) = .ok coord := by
  have consumedEq : consumed = (encodeBytes source).size :=
    derivation.lexing.consumes_rep (ByteRep.encodeBytes source)
  calc
    Decode.decodeImportedCoordCeiling (encodeBytes source) =
        (Decode.tokenize (encodeBytes source)).bind fun result =>
          if result.2 = (encodeBytes source).size then
            (Decode.scanSelected (result.1.length + 1) result.1 ["COORD_CEILING"] []).bind
              fun selected => match selected.1 with
                | [] => Except.error "COORD_CEILING is missing or duplicated"
                | decl :: [] => Except.ok decl
                | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated"
          else Except.error "import tokenizer did not consume every source byte" := rfl
    _ = (Except.ok (tokens, consumed) : Decode.Result (List Token × Nat)).bind fun result =>
          if result.2 = (encodeBytes source).size then
            (Decode.scanSelected (result.1.length + 1) result.1 ["COORD_CEILING"] []).bind
              fun selected => match selected.1 with
                | [] => Except.error "COORD_CEILING is missing or duplicated"
                | decl :: [] => Except.ok decl
                | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated"
          else Except.error "import tokenizer did not consume every source byte" :=
      congrArg
        (fun decoded => decoded.bind fun result =>
          if result.2 = (encodeBytes source).size then
            (Decode.scanSelected (result.1.length + 1) result.1 ["COORD_CEILING"] []).bind
              fun selected => match selected.1 with
                | [] => Except.error "COORD_CEILING is missing or duplicated"
                | decl :: [] => Except.ok decl
                | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated"
          else Except.error "import tokenizer did not consume every source byte")
        derivation.lexing.sound
    _ = if consumed = (encodeBytes source).size then
          (Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] []).bind
            fun selected => match selected.1 with
              | [] => Except.error "COORD_CEILING is missing or duplicated"
              | decl :: [] => Except.ok decl
              | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated"
        else Except.error "import tokenizer did not consume every source byte" := rfl
    _ = (Decode.scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] []).bind
          fun selected => match selected.1 with
            | [] => Except.error "COORD_CEILING is missing or duplicated"
            | decl :: [] => Except.ok decl
            | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated" :=
      if_pos consumedEq
    _ = (Except.ok ([coord], []) : Decode.Result (List ConstantDecl × List FunctionDecl)).bind
          fun selected => match selected.1 with
            | [] => Except.error "COORD_CEILING is missing or duplicated"
            | decl :: [] => Except.ok decl
            | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated" :=
      congrArg
        (fun selected => selected.bind fun result => match result.1 with
          | [] => Except.error "COORD_CEILING is missing or duplicated"
          | decl :: [] => Except.ok decl
          | _ :: _ :: _ => Except.error "COORD_CEILING is missing or duplicated")
        derivation.scanned
    _ = Except.ok coord := rfl

end ImportedCoordDerivation

def composeDecoded (decoded : CertifiedDecoded) (coord : ConstantDecl) : CertifiedDecoded :=
  Decode.appendImportedCoord decoded coord

inductive DecodeDerivation (table std : SourceBytes) (value : CertifiedDecoded) : Prop where
  | intro (decoded : CertifiedDecoded) (coord : ConstantDecl)
      (tableDecodes : Decode.decodeTable (encodeBytes table) = .ok decoded)
      (coordDecodes : Decode.decodeImportedCoordCeiling (encodeBytes std) = .ok coord)
      (coordValue : coord.value = .literal 8796093022208)
      (composed : value = composeDecoded decoded coord) : DecodeDerivation table std value

theorem decodeDerivation_of_grammar {table std : SourceBytes} {tableTokens stdTokens : List Token}
    {tableConsumed stdConsumed : Nat} {decoded value : CertifiedDecoded} {coord : ConstantDecl}
    (tableDerivation : GrammarDerivation table tableTokens tableConsumed decoded)
    (stdDerivation : ImportedCoordDerivation std stdTokens stdConsumed coord)
    (coordValue : coord.value = .literal 8796093022208)
    (composed : value = composeDecoded decoded coord) : DecodeDerivation table std value :=
  ⟨decoded, coord, tableDerivation.sound, stdDerivation.sound, coordValue, composed⟩

private def composeImported (decoded : CertifiedDecoded) (coord : ConstantDecl) :
    Decode.Result CertifiedDecoded :=
  match coord.value with
  | .literal value =>
      match Nat.beq value 8796093022208 with
      | true => .ok (composeDecoded decoded coord)
      | false => .error "COORD_CEILING has the wrong value"
  | .name _ => .error "COORD_CEILING has the wrong value"
  | .call _ _ => .error "COORD_CEILING has the wrong value"
  | .memory _ => .error "COORD_CEILING has the wrong value"
  | .binary _ _ _ => .error "COORD_CEILING has the wrong value"
  | .conditional _ _ _ => .error "COORD_CEILING has the wrong value"

private theorem decodeDerivationFields_sound {table std : SourceBytes} {value : CertifiedDecoded}
    (decoded : CertifiedDecoded) (coord : ConstantDecl)
    (tableDecodes : Decode.decodeTable (encodeBytes table) = .ok decoded)
    (coordDecodes : Decode.decodeImportedCoordCeiling (encodeBytes std) = .ok coord)
    (coordValue : coord.value = .literal 8796093022208)
    (composed : value = composeDecoded decoded coord) :
    Decode.decodeClosure (encodeBytes table) (encodeBytes std) = .ok value := by
  calc
    Decode.decodeClosure (encodeBytes table) (encodeBytes std) =
        (Decode.decodeTable (encodeBytes table)).bind fun decoded =>
          (Decode.decodeImportedCoordCeiling (encodeBytes std)).bind (composeImported decoded) := rfl
    _ = (Except.ok decoded : Decode.Result CertifiedDecoded).bind fun decoded =>
          (Decode.decodeImportedCoordCeiling (encodeBytes std)).bind (composeImported decoded) :=
      congrArg
        (fun result => result.bind fun decoded =>
          (Decode.decodeImportedCoordCeiling (encodeBytes std)).bind (composeImported decoded))
        tableDecodes
    _ = (Decode.decodeImportedCoordCeiling (encodeBytes std)).bind (composeImported decoded) := rfl
    _ = (Except.ok coord : Decode.Result ConstantDecl).bind (composeImported decoded) :=
      congrArg (fun result => result.bind (composeImported decoded)) coordDecodes
    _ = composeImported decoded coord := rfl
    _ = match Expr.literal 8796093022208 with
        | .literal coordinate =>
            match Nat.beq coordinate 8796093022208 with
            | true => Except.ok (composeDecoded decoded coord)
            | false => Except.error "COORD_CEILING has the wrong value"
        | .name _ => Except.error "COORD_CEILING has the wrong value"
        | .call _ _ => Except.error "COORD_CEILING has the wrong value"
        | .memory _ => Except.error "COORD_CEILING has the wrong value"
        | .binary _ _ _ => Except.error "COORD_CEILING has the wrong value"
        | .conditional _ _ _ => Except.error "COORD_CEILING has the wrong value" :=
      congrArg
        (fun coordinate : Expr => match coordinate with
          | .literal coordinate =>
              match Nat.beq coordinate 8796093022208 with
              | true => Except.ok (composeDecoded decoded coord)
              | false => Except.error "COORD_CEILING has the wrong value"
          | .name _ => Except.error "COORD_CEILING has the wrong value"
          | .call _ _ => Except.error "COORD_CEILING has the wrong value"
          | .memory _ => Except.error "COORD_CEILING has the wrong value"
          | .binary _ _ _ => Except.error "COORD_CEILING has the wrong value"
          | .conditional _ _ _ => Except.error "COORD_CEILING has the wrong value")
        coordValue
    _ = Except.ok (composeDecoded decoded coord) := rfl
    _ = Except.ok value := congrArg Except.ok composed.symm

theorem decodeDerivation_sound {table std : SourceBytes} {value : CertifiedDecoded} :
    DecodeDerivation table std value →
      Decode.decodeClosure (encodeBytes table) (encodeBytes std) = .ok value := by
  intro derivation
  cases derivation with
  | intro decoded coord tableDecodes coordDecodes coordValue composed =>
      exact decodeDerivationFields_sound decoded coord tableDecodes coordDecodes coordValue composed

namespace ParsesClosure

theorem decodeDerivation {bundle : SourceBundle} {program : Program}
    (closure : ParsesClosure bundle program) :
    ∃ entryBytes stdBytes coord value,
      bundle.fileBytes? bundle.entry = some entryBytes ∧
      bundle.fileBytes? stdPath = some stdBytes ∧
      coord.value = .literal 8796093022208 ∧
      DecodeDerivation entryBytes stdBytes value ∧
      value.val.program = { program with constants := program.constants ++ [coord] } := by
  cases closure with
  | intro entryBytes entryLookup decoded entryDecodes programEq _ _ _ stdBytes stdLookup coord
      coordDecodes coordValue =>
      refine ⟨entryBytes, stdBytes, coord, composeDecoded decoded coord, entryLookup, stdLookup,
        coordValue, ?_, ?_⟩
      · exact ⟨decoded, coord, entryDecodes, coordDecodes, coordValue, rfl⟩
      · simp only [composeDecoded, Decode.appendImportedCoord]
        rw [programEq]

theorem decodeClosure_ok {bundle : SourceBundle} {program : Program}
    (closure : ParsesClosure bundle program) :
    ∃ entryBytes stdBytes value,
      bundle.fileBytes? bundle.entry = some entryBytes ∧
      bundle.fileBytes? stdPath = some stdBytes ∧
      Decode.decodeClosure (encodeBytes entryBytes) (encodeBytes stdBytes) = .ok value := by
  obtain ⟨entryBytes, stdBytes, _, value, entryLookup, stdLookup, _, derivation, _⟩ :=
    closure.decodeDerivation
  exact ⟨entryBytes, stdBytes, value, entryLookup, stdLookup,
    decodeDerivation_sound derivation⟩

end ParsesClosure

end Slang.Source
