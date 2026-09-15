import Slang.Audit.Empty
import Slang.Derivation.Decode
import Slang.Derivation.Lexing
import Slang.Derivation.Grammar
import Slang.Derivation.Statements
import Slang.Derivation.Composition
import Slang.Derivation.Selection
import Slang.Derivation.Validation

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Derivation.DecodeTrace

#audit_empty Slang.Derivation.Derives.ask_done : ∀ {Request : Type} {Answer : Request → Type} {body : (request : Request) → Layer Request Answer (Answer request)} {request : Request} {answer : Answer request}, Derives body (body request) answer → Derives body (.ask request fun result => .done result) answer
#audit_empty Slang.Derivation.Derives.done_eq : type_of% @Slang.Derivation.Derives.done_eq

#audit_empty Slang.Derivation.DecodeTrace.evaluate : (q : Slang.Derivation.DecodeTrace.Request) → Slang.Derivation.DecodeTrace.Answer q
#audit_empty Slang.Derivation.DecodeTrace.body_eq : ∀ (q : Slang.Derivation.DecodeTrace.Request), Slang.Derivation.DecodeTrace.evaluate q = Slang.Derivation.fold Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.body q)
#audit_empty Slang.Derivation.DecodeTrace.Run.sound : ∀ {q : Slang.Derivation.DecodeTrace.Request} {answer : Slang.Derivation.DecodeTrace.Answer q}, Slang.Derivation.DecodeTrace.Run q answer → Slang.Derivation.DecodeTrace.evaluate q = answer
#audit_empty Slang.Derivation.DecodeTrace.multibyte_run : Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.ok ['é'])
#audit_empty Slang.Derivation.DecodeTrace.multibyte_result : Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) = Except.ok ['é']
#audit_empty Slang.Derivation.DecodeTrace.multibyte_wrong : ¬Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.error "source is not UTF-8")

-- The embedding adapters delegate a `Lexing`/`Selection`/`Validation` `Run` into
-- the matching `DecodeTrace` request; each transports the child's checked
-- equality through `Derives.done_eq` on the `.done` leaf of `DecodeTrace.body`.
#audit_empty Slang.Derivation.DecodeTrace.lexing_of_run : type_of% @Slang.Derivation.DecodeTrace.lexing_of_run
#audit_empty Slang.Derivation.DecodeTrace.selection_of_run : type_of% @Slang.Derivation.DecodeTrace.selection_of_run
#audit_empty Slang.Derivation.DecodeTrace.validation_of_run : type_of% @Slang.Derivation.DecodeTrace.validation_of_run

/-- Fixture the zero-axiom runner mutates to prove the command rejects a nonempty cone. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.Derivation.DecodeTrace.nat_add_zero : (0 : Nat) + 0 = 0

#audit_empty Slang.Derivation.Lexing.evaluate : (q : Slang.Derivation.Lexing.Request) → Slang.Derivation.Lexing.Answer q
#audit_empty Slang.Derivation.Lexing.lexAsk : Nat → List Char → Nat → List Slang.Decode.Token → Slang.Derivation.Layer Slang.Derivation.Lexing.Request Slang.Derivation.Lexing.Answer (Slang.Decode.Result (List Slang.Decode.Token × Nat))
#audit_empty Slang.Derivation.Lexing.body_eq : ∀ (q : Slang.Derivation.Lexing.Request), Slang.Derivation.Lexing.evaluate q = Slang.Derivation.fold Slang.Derivation.Lexing.evaluate (Slang.Derivation.Lexing.body q)
#audit_empty Slang.Derivation.Lexing.Run.sound : ∀ {q : Slang.Derivation.Lexing.Request} {answer : Slang.Derivation.Lexing.Answer q}, Slang.Derivation.Lexing.Run q answer → Slang.Derivation.Lexing.evaluate q = answer
#audit_empty Slang.Derivation.Lexing.lexChars_run : ∀ (fuel : Nat) (chars : List Char) (pos : Nat) (tokens : List Slang.Decode.Token), Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.lex fuel chars pos tokens) (Slang.Decode.lexChars fuel chars pos tokens)
#audit_empty Slang.Derivation.Lexing.tokenize_run : ∀ (bytes : ByteArray), Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.tokenize bytes) (Slang.Decode.tokenize bytes)

end Slang.Derivation.DecodeTrace

-- The parser-derivation headlines. Each name is bound to its own elaborated
-- type via `type_of%`; the audited property is the empty axiom cone.
#audit_empty Slang.Derivation.Grammar.evaluate : type_of% @Slang.Derivation.Grammar.evaluate
#audit_empty Slang.Derivation.Grammar.body_eq : type_of% @Slang.Derivation.Grammar.body_eq
#audit_empty Slang.Derivation.Grammar.Run.sound : type_of% @Slang.Derivation.Grammar.Run.sound
#audit_empty Slang.Derivation.Grammar.expr_run : type_of% @Slang.Derivation.Grammar.expr_run
#audit_empty Slang.Derivation.Grammar.primary_run : type_of% @Slang.Derivation.Grammar.primary_run
#audit_empty Slang.Derivation.Grammar.args_run : type_of% @Slang.Derivation.Grammar.args_run
#audit_empty Slang.Derivation.Grammar.binary_run : type_of% @Slang.Derivation.Grammar.binary_run
#audit_empty Slang.Derivation.Statements.evaluate : type_of% @Slang.Derivation.Statements.evaluate
#audit_empty Slang.Derivation.Statements.body_eq : type_of% @Slang.Derivation.Statements.body_eq
#audit_empty Slang.Derivation.Statements.Run.sound : type_of% @Slang.Derivation.Statements.Run.sound
#audit_empty Slang.Derivation.Statements.block_run : type_of% @Slang.Derivation.Statements.block_run
#audit_empty Slang.Derivation.Statements.stmt_run : type_of% @Slang.Derivation.Statements.stmt_run

-- The cross-family transfer adapters and their composed fixtures. The adapters
-- transport a child `Run` into the parent `.done` leaf via
-- `Slang.Derivation.Derives.done_eq`; the negative controls reject a wrong answer.
#audit_empty Slang.Derivation.Composition.statements_of_grammar_expr : type_of% @Slang.Derivation.Composition.statements_of_grammar_expr
#audit_empty Slang.Derivation.Composition.lexing_of_decodetrace_chars : type_of% @Slang.Derivation.Composition.lexing_of_decodetrace_chars
#audit_empty Slang.Derivation.Composition.grammar_expr_zero_run : type_of% @Slang.Derivation.Composition.grammar_expr_zero_run
#audit_empty Slang.Derivation.Composition.statements_expr_zero_run : type_of% @Slang.Derivation.Composition.statements_expr_zero_run
#audit_empty Slang.Derivation.Composition.statements_expr_zero_wrong : type_of% @Slang.Derivation.Composition.statements_expr_zero_wrong
#audit_empty Slang.Derivation.Composition.lexing_chars_multibyte_run : type_of% @Slang.Derivation.Composition.lexing_chars_multibyte_run
#audit_empty Slang.Derivation.Composition.lexing_chars_multibyte_result : type_of% @Slang.Derivation.Composition.lexing_chars_multibyte_result
#audit_empty Slang.Derivation.Composition.lexing_chars_multibyte_wrong : type_of% @Slang.Derivation.Composition.lexing_chars_multibyte_wrong

-- The declaration-scanner derivation family. The `expr` and `block` requests are
-- external-call leaves; the adapters transport a `Grammar`/`Statements` `Run`
-- into that leaf via `Derives.done_eq`, and the wrong-answer controls reject a
-- non-derivable answer through the composed path.
#audit_empty Slang.Derivation.Selection.evaluate : type_of% @Slang.Derivation.Selection.evaluate
#audit_empty Slang.Derivation.Selection.scan_unfold : type_of% @Slang.Derivation.Selection.scan_unfold
#audit_empty Slang.Derivation.Selection.body_eq : type_of% @Slang.Derivation.Selection.body_eq
#audit_empty Slang.Derivation.Selection.Run.sound : type_of% @Slang.Derivation.Selection.Run.sound
#audit_empty Slang.Derivation.Selection.parameters_run : type_of% @Slang.Derivation.Selection.parameters_run
#audit_empty Slang.Derivation.Selection.line_run : type_of% @Slang.Derivation.Selection.line_run
#audit_empty Slang.Derivation.Selection.function_run : type_of% @Slang.Derivation.Selection.function_run
#audit_empty Slang.Derivation.Selection.scan_run : type_of% @Slang.Derivation.Selection.scan_run
#audit_empty Slang.Derivation.Selection.expr_of_grammar : type_of% @Slang.Derivation.Selection.expr_of_grammar
#audit_empty Slang.Derivation.Selection.block_of_statements : type_of% @Slang.Derivation.Selection.block_of_statements
#audit_empty Slang.Derivation.Selection.expr_zero_run : type_of% @Slang.Derivation.Selection.expr_zero_run
#audit_empty Slang.Derivation.Selection.block_zero_run : type_of% @Slang.Derivation.Selection.block_zero_run
#audit_empty Slang.Derivation.Selection.expr_zero_wrong : type_of% @Slang.Derivation.Selection.expr_zero_wrong
#audit_empty Slang.Derivation.Selection.block_zero_wrong : type_of% @Slang.Derivation.Selection.block_zero_wrong

-- The boolean-validation derivation family over the `Decode.decodeTokens`
-- validators. Every request is a `.done` leaf or a `.ask` chain of recursive
-- requests; the leaf validators `distinct`, `spansValid`, `resolveCalls` and
-- `resolveNames` are single calls, and the recursive validators are covered by
-- their `_run` completeness lemmas.
#audit_empty Slang.Derivation.Validation.evaluate : type_of% @Slang.Derivation.Validation.evaluate
#audit_empty Slang.Derivation.Validation.body_eq : type_of% @Slang.Derivation.Validation.body_eq
#audit_empty Slang.Derivation.Validation.Run.sound : type_of% @Slang.Derivation.Validation.Run.sound
#audit_empty Slang.Derivation.Validation.delimiters_run : type_of% @Slang.Derivation.Validation.delimiters_run
#audit_empty Slang.Derivation.Validation.distinct_run : type_of% @Slang.Derivation.Validation.distinct_run
#audit_empty Slang.Derivation.Validation.spans_run : type_of% @Slang.Derivation.Validation.spans_run
#audit_empty Slang.Derivation.Validation.expressionCalls_run : type_of% @Slang.Derivation.Validation.expressionCalls_run
#audit_empty Slang.Derivation.Validation.expressionsCalls_run : type_of% @Slang.Derivation.Validation.expressionsCalls_run
#audit_empty Slang.Derivation.Validation.statementCalls_run : type_of% @Slang.Derivation.Validation.statementCalls_run
#audit_empty Slang.Derivation.Validation.statementsCalls_run : type_of% @Slang.Derivation.Validation.statementsCalls_run
#audit_empty Slang.Derivation.Validation.resolveCalls_run : type_of% @Slang.Derivation.Validation.resolveCalls_run
#audit_empty Slang.Derivation.Validation.expressionNames_run : type_of% @Slang.Derivation.Validation.expressionNames_run
#audit_empty Slang.Derivation.Validation.expressionsNames_run : type_of% @Slang.Derivation.Validation.expressionsNames_run
#audit_empty Slang.Derivation.Validation.statementsNames_run : type_of% @Slang.Derivation.Validation.statementsNames_run
#audit_empty Slang.Derivation.Validation.resolveNames_run : type_of% @Slang.Derivation.Validation.resolveNames_run
