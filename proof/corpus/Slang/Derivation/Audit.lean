import Slang.Audit.Empty
import Slang.Derivation.Decode
import Slang.Derivation.Lexing
import Slang.Derivation.Grammar
import Slang.Derivation.Statements

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
