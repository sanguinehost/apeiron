import Slang.Derivation.Steps
import Slang.Derivation.Decode
import Slang.Derivation.Lexing
import Slang.Derivation.Grammar
import Slang.Derivation.Statements

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Derivation.Composition

open Slang Slang.Decode

/-- Transfer a `Grammar` expression derivation to the `Statements` family.

`Statements.body (.expr fuel tokens)` is the leaf `.done (Decode.parseExprFuel fuel 0 tokens)`,
the same real call that `Grammar.evaluate (.expr fuel 0 tokens)` names. The child's checked
equality, `Grammar.Run.sound`, is transported into the parent `.done` by
`Slang.Derivation.Derives.done_eq`; no parser evaluation and no new axiom. -/
theorem statements_of_grammar_expr {fuel : Nat} {tokens : List Token}
    {answer : Result (Expr × List Token)}
    (derivation : Grammar.Run (.expr fuel 0 tokens) answer) :
    Statements.Run (.expr fuel tokens) answer :=
  Slang.Derivation.Derives.done_eq (Grammar.Run.sound derivation)

/-- Transfer a `DecodeTrace` byte-level derivation to the `Lexing` character request.

`Lexing.body (.chars bytes)` is the leaf `.done (Decode.decodeUtf8Chars bytes)`, and
`Decode.decodeUtf8Chars bytes = Decode.decodeUtf8CharsFuel (bytes.length + 1) bytes`, the real
call that `DecodeTrace.evaluate (.utf8Fuel (bytes.length + 1) bytes)` names. -/
theorem lexing_of_decodetrace_chars {bytes : List UInt8} {answer : Result (List Char)}
    (derivation : DecodeTrace.Run (.utf8Fuel (bytes.length + 1) bytes) answer) :
    Lexing.Run (.chars bytes) answer :=
  Slang.Derivation.Derives.done_eq (DecodeTrace.Run.sound derivation)

/-- The concrete `Grammar` leaf at fuel zero: the parser reports its nesting limit. -/
theorem grammar_expr_zero_run :
    Grammar.Run (.expr 0 0 ([] : List Token))
      (Except.error "expression nesting limit reached") :=
  Slang.Derivation.Derives.done _

/-- The composed `Statements` derivation for that leaf, through the adapter. -/
theorem statements_expr_zero_run :
    Statements.Run (.expr 0 ([] : List Token))
      (Except.error "expression nesting limit reached") :=
  statements_of_grammar_expr grammar_expr_zero_run

/-- A wrong answer is not derivable through the composed `Grammar` -> `Statements` path. -/
theorem statements_expr_zero_wrong :
    ¬ Statements.Run (.expr 0 ([] : List Token))
      (Except.ok (Expr.literal 0, [])) := by
  intro derivation
  have sound := Statements.Run.sound derivation
  have limit : Statements.evaluate (.expr 0 ([] : List Token)) =
      Except.error "expression nesting limit reached" := rfl
  rw [limit] at sound
  cases sound

/-- The composed `Lexing` derivation for the U+00E9 byte fixture of `DecodeTrace`. -/
theorem lexing_chars_multibyte_run :
    Lexing.Run (.chars DecodeTrace.multibyteBytes) (Except.ok ['é']) :=
  lexing_of_decodetrace_chars DecodeTrace.multibyte_run

/-- The composed path names the real character-decoder result. -/
theorem lexing_chars_multibyte_result :
    Lexing.evaluate (.chars DecodeTrace.multibyteBytes) = Except.ok ['é'] :=
  Lexing.Run.sound lexing_chars_multibyte_run

/-- A wrong answer for the U+00E9 fixture is not derivable through the composed path. -/
theorem lexing_chars_multibyte_wrong :
    ¬ Lexing.Run (.chars DecodeTrace.multibyteBytes)
      (Except.error "source is not UTF-8") := by
  intro derivation
  have sound := Lexing.Run.sound derivation
  rw [lexing_chars_multibyte_result] at sound
  cases sound

end Slang.Derivation.Composition
