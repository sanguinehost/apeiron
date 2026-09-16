import Slang.Source.Parsing.Selection
import Slang.Source.Parsing.Validation

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Parsing

open Slang Slang.Decode

private def acceptedDecoded (bytes : ByteArray) (consumed : Nat)
    (constants : List ConstantDecl) (functions : List FunctionDecl) : CertifiedDecoded :=
  let program : Program :=
    { constants := constants, functions := functions, sourceBytes := bytes.size,
      consumedBytes := consumed }
  let certificate : Certificate := ⟨true, true, true, true, true⟩
  ⟨{ program := program, certificate := certificate }, ⟨rfl, rfl, rfl, rfl, rfl⟩⟩

/-- The accepted decoder layers and guards, with their common selected declarations. -/
structure DecodeTokensDerivation (bytes : ByteArray) (tokens : List Token) (consumed : Nat)
    (decoded : CertifiedDecoded) : Prop where
  exactConsumption : (consumed != bytes.size) = false
  delimiters : DelimitersBalancedDerivation tokens []
  selected : ScanSelected (tokens.length + 1) tokens selectedConstantNames selectedFunctionNames
    decoded.val.program.constants decoded.val.program.functions
  constantCount : ((decoded.val.program.constants.map (·.name)).length != 8) = false
  functionCount : ((decoded.val.program.functions.map (·.name)).length !=
    selectedFunctionNames.length) = false
  constantNames : DistinctDerivation
    (decoded.val.program.constants.map (·.name) ++ ["COORD_CEILING"])
  functionNames : DistinctDerivation (decoded.val.program.functions.map (·.name))
  spans : SpansValidDerivation bytes.size
    (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span))
  calls : CallsResolveDerivation decoded.val.program.functions
  names : NamesResolveDerivation (decoded.val.program.constants ++ [{
    name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }])
    decoded.val.program.functions
  result : decoded = acceptedDecoded bytes consumed decoded.val.program.constants
    decoded.val.program.functions

namespace DecodeTokensDerivation

/-- Accepted structural layers entail the executable token decoder result. -/
theorem sound {bytes : ByteArray} {tokens : List Token} {consumed : Nat}
    {decoded : CertifiedDecoded} (derivation : DecodeTokensDerivation bytes tokens consumed decoded) :
    Decode.decodeTokens bytes tokens consumed = .ok decoded := by
  unfold Decode.decodeTokens
  rw [derivation.exactConsumption, if_neg Bool.false_ne_true]
  rw [derivation.delimiters.sound]
  change (if false = true then _ else _) = _
  rw [if_neg Bool.false_ne_true]
  rw [derivation.selected.sound]
  change (if ((decoded.val.program.constants.map (·.name)).length != 8) = true then _ else _) = _
  rw [derivation.constantCount, if_neg Bool.false_ne_true]
  rw [derivation.functionCount, if_neg Bool.false_ne_true]
  rw [derivation.constantNames.sound, derivation.functionNames.sound]
  change (if false = true then _ else _) = _
  rw [if_neg Bool.false_ne_true]
  rw [derivation.spans.sound]
  change (if false = true then _ else _) = _
  rw [if_neg Bool.false_ne_true]
  rw [derivation.calls.sound]
  change (if false = true then _ else _) = _
  rw [if_neg Bool.false_ne_true]
  rw [derivation.names.sound]
  change (if false = true then _ else _) = _
  rw [if_neg Bool.false_ne_true]
  rw [derivation.result]
  unfold acceptedDecoded
  rfl

end DecodeTokensDerivation

end Slang.Source.Parsing
