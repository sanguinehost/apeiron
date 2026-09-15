import Slang.Audit.Empty
import Slang.Derivation.Steps
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 4096

namespace Slang.Derivation.Validation

open Slang Slang.Decode

/-- Requests name the boolean validators that `Decode.decodeTokens` consults. -/
inductive Request where
  | delimiters (tokens : List Token) (stack : List TokenKind)
  | distinct (names : List String)
  | spans (size : Nat) (spans : List Span)
  | expressionCalls (available : List String) (expression : Expr)
  | expressionsCalls (available : List String) (expressions : List Expr)
  | statementCalls (available : List String) (statement : Stmt)
  | statementsCalls (available : List String) (statements : List Stmt)
  | resolveCalls (functions : List FunctionDecl)
  | expressionNames (available : List String) (expression : Expr)
  | expressionsNames (available : List String) (expressions : List Expr)
  | statementsNames (fuel : Nat) (available : List String) (statements : List Stmt)
  | resolveNames (constants : List ConstantDecl) (functions : List FunctionDecl)

/-- Every validator answers a boolean. -/
abbrev Answer : Request → Type := fun _ => Bool

/-- The real validator, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .delimiters tokens stack => Decode.delimitersBalanced tokens stack
  | .distinct names => Decode.distinct names
  | .spans size spans => Decode.spansValid size spans
  | .expressionCalls available expression => Decode.expressionCallsResolve available expression
  | .expressionsCalls available expressions => Decode.expressionsCallsResolve available expressions
  | .statementCalls available statement => Decode.statementCallsResolve available statement
  | .statementsCalls available statements => Decode.statementsCallsResolve available statements
  | .resolveCalls functions => Decode.resolveCalls functions
  | .expressionNames available expression => Decode.expressionNamesResolve available expression
  | .expressionsNames available expressions => Decode.expressionsNamesResolve available expressions
  | .statementsNames fuel available statements =>
      Decode.statementsNamesResolve fuel available statements
  | .resolveNames constants functions => Decode.resolveNames constants functions

/-- One unfolding of each validator, exposing recursive calls as requests.

The structural recursion is exposed with `.ask`; the non-structural validators
`distinct`, `spansValid`, `resolveCalls` and `resolveNames` are `.done` leaves of
the real call, because their iteration is `List.all`/`List.eraseDups` rather than
a recursive `Decode` declaration. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .delimiters [] stack =>
      match stack with
      | [] => .done true
      | _ :: _ => .done false
  | .delimiters (token :: rest) stack =>
      match token.kind with
      | .lparen => .ask (.delimiters rest (.lparen :: stack)) (fun answer => .done answer)
      | .lbrace => .ask (.delimiters rest (.lbrace :: stack)) (fun answer => .done answer)
      | .lbracket => .ask (.delimiters rest (.lbracket :: stack)) (fun answer => .done answer)
      | .rparen =>
          match stack with
          | [] => .done false
          | top :: tail =>
              if top = .lparen then
                .ask (.delimiters rest tail) (fun answer => .done answer)
              else .done false
      | .rbrace =>
          match stack with
          | [] => .done false
          | top :: tail =>
              if top = .lbrace then
                .ask (.delimiters rest tail) (fun answer => .done answer)
              else .done false
      | .rbracket =>
          match stack with
          | [] => .done false
          | top :: tail =>
              if top = .lbracket then
                .ask (.delimiters rest tail) (fun answer => .done answer)
              else .done false
      | .ident _ => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .numeral _ => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .comma => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .semi => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .newline => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .assign => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .lt => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .le => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .gt => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .ge => .ask (.delimiters rest stack) (fun answer => .done answer)
      | .other _ => .ask (.delimiters rest stack) (fun answer => .done answer)
  | .distinct names => .done (Decode.distinct names)
  | .spans size spans => .done (Decode.spansValid size spans)
  | .expressionCalls available expression =>
      match expression with
      | .literal _ => .done true
      | .name _ => .done true
      | .call callee args =>
          .ask (.expressionsCalls available args)
            (fun answer => .done (available.contains callee && answer))
      | .memory address =>
          .ask (.expressionCalls available address) (fun answer => .done answer)
      | .binary _ left right =>
          .ask (.expressionCalls available left) (fun leftAnswer =>
            .ask (.expressionCalls available right) (fun rightAnswer =>
              .done (leftAnswer && rightAnswer)))
      | .conditional thenValue condition elseValue =>
          .ask (.expressionCalls available thenValue) (fun thenAnswer =>
            .ask (.expressionCalls available condition) (fun conditionAnswer =>
              .ask (.expressionCalls available elseValue) (fun elseAnswer =>
                .done (thenAnswer && conditionAnswer && elseAnswer))))
  | .expressionsCalls available expressions =>
      match expressions with
      | [] => .done true
      | expression :: rest =>
          .ask (.expressionCalls available expression) (fun headAnswer =>
            .ask (.expressionsCalls available rest) (fun tailAnswer =>
              .done (headAnswer && tailAnswer)))
  | .statementCalls available statement =>
      match statement with
      | .bind _ value =>
          .ask (.expressionCalls available value) (fun answer => .done answer)
      | .store address value =>
          .ask (.expressionCalls available address) (fun addressAnswer =>
            .ask (.expressionCalls available value) (fun valueAnswer =>
              .done (addressAnswer && valueAnswer)))
      | .return_ value =>
          .ask (.expressionCalls available value) (fun answer => .done answer)
      | .expression value =>
          .ask (.expressionCalls available value) (fun answer => .done answer)
      | .if_ condition thenBody elseBody =>
          .ask (.expressionCalls available condition) (fun conditionAnswer =>
            .ask (.statementsCalls available thenBody) (fun thenAnswer =>
              match elseBody with
              | none => .done (conditionAnswer && thenAnswer && true)
              | some body =>
                  .ask (.statementsCalls available body) (fun elseAnswer =>
                    .done (conditionAnswer && thenAnswer && elseAnswer))))
  | .statementsCalls available statements =>
      match statements with
      | [] => .done true
      | statement :: rest =>
          .ask (.statementCalls available statement) (fun headAnswer =>
            .ask (.statementsCalls available rest) (fun tailAnswer =>
              .done (headAnswer && tailAnswer)))
  | .resolveCalls functions => .done (Decode.resolveCalls functions)
  | .expressionNames available expression =>
      match expression with
      | .literal _ => .done true
      | .name name => .done (available.contains name)
      | .call _ args =>
          .ask (.expressionsNames available args) (fun answer => .done answer)
      | .memory address =>
          .ask (.expressionNames available address) (fun answer => .done answer)
      | .binary _ left right =>
          .ask (.expressionNames available left) (fun leftAnswer =>
            .ask (.expressionNames available right) (fun rightAnswer =>
              .done (leftAnswer && rightAnswer)))
      | .conditional thenValue condition elseValue =>
          .ask (.expressionNames available thenValue) (fun thenAnswer =>
            .ask (.expressionNames available condition) (fun conditionAnswer =>
              .ask (.expressionNames available elseValue) (fun elseAnswer =>
                .done (thenAnswer && conditionAnswer && elseAnswer))))
  | .expressionsNames available expressions =>
      match expressions with
      | [] => .done true
      | expression :: rest =>
          .ask (.expressionNames available expression) (fun headAnswer =>
            .ask (.expressionsNames available rest) (fun tailAnswer =>
              .done (headAnswer && tailAnswer)))
  | .statementsNames 0 _ _ => .done false
  | .statementsNames _ _ [] => .done true
  | .statementsNames (fuel + 1) available (statement :: rest) =>
      match statement with
      | .bind name value =>
          .ask (.expressionNames available value) (fun valueAnswer =>
            .ask (.statementsNames fuel (name :: available) rest) (fun restAnswer =>
              .done (valueAnswer && restAnswer)))
      | .store address value =>
          .ask (.expressionNames available address) (fun addressAnswer =>
            .ask (.expressionNames available value) (fun valueAnswer =>
              .ask (.statementsNames fuel available rest) (fun restAnswer =>
                .done (addressAnswer && valueAnswer && restAnswer))))
      | .return_ value =>
          .ask (.expressionNames available value) (fun valueAnswer =>
            .ask (.statementsNames fuel available rest) (fun restAnswer =>
              .done (valueAnswer && restAnswer)))
      | .expression value =>
          .ask (.expressionNames available value) (fun valueAnswer =>
            .ask (.statementsNames fuel available rest) (fun restAnswer =>
              .done (valueAnswer && restAnswer)))
      | .if_ condition thenBody elseBody =>
          .ask (.expressionNames available condition) (fun conditionAnswer =>
            .ask (.statementsNames fuel available thenBody) (fun thenAnswer =>
              match elseBody with
              | none =>
                  .ask (.statementsNames fuel available rest) (fun restAnswer =>
                    .done (conditionAnswer && thenAnswer && true && restAnswer))
              | some body =>
                  .ask (.statementsNames fuel available body) (fun elseAnswer =>
                    .ask (.statementsNames fuel available rest) (fun restAnswer =>
                      .done (conditionAnswer && thenAnswer && elseAnswer && restAnswer)))))
  | .resolveNames constants functions => .done (Decode.resolveNames constants functions)

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .delimiters tokens stack => by
      cases tokens with
      | nil =>
          dsimp only [evaluate, body, fold, Decode.delimitersBalanced]
          cases stack <;> rfl
      | cons token rest =>
          dsimp only [evaluate, body, fold, Decode.delimitersBalanced]
          cases hk : token.kind
          case lparen => rfl
          case lbrace => rfl
          case lbracket => rfl
          case rparen =>
              cases stack with
              | nil =>
                  dsimp only []
                  rfl
              | cons top tail =>
                  dsimp only []
                  by_cases h : top = .lparen
                  · conv => lhs; rw [if_pos h]
                    conv => rhs; rw [if_pos h]
                    rfl
                  · conv => lhs; rw [if_neg h]
                    conv => rhs; rw [if_neg h]
                    rfl
          case rbrace =>
              cases stack with
              | nil =>
                  dsimp only []
                  rfl
              | cons top tail =>
                  dsimp only []
                  by_cases h : top = .lbrace
                  · conv => lhs; rw [if_pos h]
                    conv => rhs; rw [if_pos h]
                    rfl
                  · conv => lhs; rw [if_neg h]
                    conv => rhs; rw [if_neg h]
                    rfl
          case rbracket =>
              cases stack with
              | nil =>
                  dsimp only []
                  rfl
              | cons top tail =>
                  dsimp only []
                  by_cases h : top = .lbracket
                  · conv => lhs; rw [if_pos h]
                    conv => rhs; rw [if_pos h]
                    rfl
                  · conv => lhs; rw [if_neg h]
                    conv => rhs; rw [if_neg h]
                    rfl
          all_goals rfl
  | .distinct names => rfl
  | .spans size spans => rfl
  | .expressionCalls available expression => by
      cases expression with
      | literal value => rfl
      | name value => rfl
      | call callee args => rfl
      | memory address => rfl
      | binary op left right => rfl
      | conditional thenValue condition elseValue => rfl
  | .expressionsCalls available expressions => by
      cases expressions with
      | nil => rfl
      | cons expression rest => rfl
  | .statementCalls available statement => by
      cases statement with
      | bind name value => rfl
      | store address value => rfl
      | return_ value => rfl
      | expression value => rfl
      | if_ condition thenBody elseBody =>
          cases elseBody with
          | none => rfl
          | some body => rfl
  | .statementsCalls available statements => by
      cases statements with
      | nil => rfl
      | cons statement rest => rfl
  | .resolveCalls functions => rfl
  | .expressionNames available expression => by
      cases expression with
      | literal value => rfl
      | name value => rfl
      | call callee args => rfl
      | memory address => rfl
      | binary op left right => rfl
      | conditional thenValue condition elseValue => rfl
  | .expressionsNames available expressions => by
      cases expressions with
      | nil => rfl
      | cons expression rest => rfl
  | .statementsNames fuel available statements => by
      cases fuel with
      | zero => rfl
      | succ fuel =>
          cases statements with
          | nil => rfl
          | cons statement rest =>
              cases statement with
              | bind name value => rfl
              | store address value => rfl
              | return_ value => rfl
              | expression value => rfl
              | if_ condition thenBody elseBody =>
                  cases elseBody with
                  | none => rfl
                  | some body => rfl
  | .resolveNames constants functions => rfl

/-- A checked derivation of the real validator's result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

/-- Every delimiter request has a checked derivation of the real validator's answer. -/
theorem delimiters_run : ∀ (tokens : List Token) (stack : List TokenKind),
    Run (.delimiters tokens stack) (Decode.delimitersBalanced tokens stack)
  | [], stack => by
      cases stack <;> exact Derives.done _
  | token :: rest, stack => by
      unfold Slang.Derivation.Validation.Run Slang.Derivation.Validation.body
      dsimp only [Decode.delimitersBalanced]
      cases hk : token.kind
      case lparen =>
          exact Derives.ask_done (body := body) (delimiters_run rest (.lparen :: stack))
      case lbrace =>
          exact Derives.ask_done (body := body) (delimiters_run rest (.lbrace :: stack))
      case lbracket =>
          exact Derives.ask_done (body := body) (delimiters_run rest (.lbracket :: stack))
      case rparen =>
          cases stack with
          | nil =>
              dsimp only []
              exact Derives.done _
          | cons top tail =>
              dsimp only []
              by_cases h : top = .lparen
              · conv => arg 2; rw [if_pos h]
                conv => arg 3; rw [if_pos h]
                exact Derives.ask_done (body := body) (delimiters_run rest tail)
              · conv => arg 2; rw [if_neg h]
                conv => arg 3; rw [if_neg h]
                exact Derives.done _
      case rbrace =>
          cases stack with
          | nil =>
              dsimp only []
              exact Derives.done _
          | cons top tail =>
              dsimp only []
              by_cases h : top = .lbrace
              · conv => arg 2; rw [if_pos h]
                conv => arg 3; rw [if_pos h]
                exact Derives.ask_done (body := body) (delimiters_run rest tail)
              · conv => arg 2; rw [if_neg h]
                conv => arg 3; rw [if_neg h]
                exact Derives.done _
      case rbracket =>
          cases stack with
          | nil =>
              dsimp only []
              exact Derives.done _
          | cons top tail =>
              dsimp only []
              by_cases h : top = .lbracket
              · conv => arg 2; rw [if_pos h]
                conv => arg 3; rw [if_pos h]
                exact Derives.ask_done (body := body) (delimiters_run rest tail)
              · conv => arg 2; rw [if_neg h]
                conv => arg 3; rw [if_neg h]
                exact Derives.done _
      all_goals
        exact Derives.ask_done (body := body) (delimiters_run rest stack)

/-- Every distinctness request is a checked leaf of the real validator. -/
theorem distinct_run (names : List String) :
    Run (.distinct names) (Decode.distinct names) :=
  Derives.done _

/-- Every span request is a checked leaf of the real validator. -/
theorem spans_run (size : Nat) (spans : List Span) :
    Run (.spans size spans) (Decode.spansValid size spans) :=
  Derives.done _

mutual
  /-- Every expression call request has a checked derivation of the real validator. -/
  theorem expressionCalls_run : ∀ (available : List String) (expression : Expr),
      Run (.expressionCalls available expression)
        (Decode.expressionCallsResolve available expression)
    | _, .literal _ => Derives.done _
    | _, .name _ => Derives.done _
    | available, .call _ args =>
        Derives.ask (body := body) (called := expressionsCalls_run available args)
          (Derives.done _)
    | available, .memory address =>
        Derives.ask_done (body := body) (expressionCalls_run available address)
    | available, .binary _ left right =>
        Derives.ask (body := body) (called := expressionCalls_run available left)
          (Derives.ask (body := body) (called := expressionCalls_run available right)
            (Derives.done _))
    | available, .conditional thenValue condition elseValue =>
        Derives.ask (body := body) (called := expressionCalls_run available thenValue)
          (Derives.ask (body := body) (called := expressionCalls_run available condition)
            (Derives.ask (body := body) (called := expressionCalls_run available elseValue)
              (Derives.done _)))

  /-- Every expression-list call request has a checked derivation of the real validator. -/
  theorem expressionsCalls_run : ∀ (available : List String) (expressions : List Expr),
      Run (.expressionsCalls available expressions)
        (Decode.expressionsCallsResolve available expressions)
    | _, [] => Derives.done _
    | available, expression :: rest =>
        Derives.ask (body := body) (called := expressionCalls_run available expression)
          (Derives.ask (body := body) (called := expressionsCalls_run available rest)
            (Derives.done _))
end

mutual
  /-- Every statement call request has a checked derivation of the real validator. -/
  theorem statementCalls_run : ∀ (available : List String) (statement : Stmt),
      Run (.statementCalls available statement)
        (Decode.statementCallsResolve available statement)
    | available, .bind _ value =>
        Derives.ask_done (body := body) (expressionCalls_run available value)
    | available, .store address value =>
        Derives.ask (body := body) (called := expressionCalls_run available address)
          (Derives.ask (body := body) (called := expressionCalls_run available value)
            (Derives.done _))
    | available, .return_ value =>
        Derives.ask_done (body := body) (expressionCalls_run available value)
    | available, .expression value =>
        Derives.ask_done (body := body) (expressionCalls_run available value)
    | available, .if_ condition thenBody elseBody =>
        Derives.ask (body := body) (called := expressionCalls_run available condition)
          (Derives.ask (body := body) (called := statementsCalls_run available thenBody)
            (match elseBody with
              | none => Derives.done _
              | some elseBody =>
                  Derives.ask (body := body) (called := statementsCalls_run available elseBody)
                    (Derives.done _)))

  /-- Every statement-list call request has a checked derivation of the real validator. -/
  theorem statementsCalls_run : ∀ (available : List String) (statements : List Stmt),
      Run (.statementsCalls available statements)
        (Decode.statementsCallsResolve available statements)
    | _, [] => Derives.done _
    | available, statement :: rest =>
        Derives.ask (body := body) (called := statementCalls_run available statement)
          (Derives.ask (body := body) (called := statementsCalls_run available rest)
            (Derives.done _))
end

/-- Every reachable-call request is a checked leaf of the real validator. -/
theorem resolveCalls_run (functions : List FunctionDecl) :
    Run (.resolveCalls functions) (Decode.resolveCalls functions) :=
  Derives.done _

mutual
  /-- Every expression name request has a checked derivation of the real validator. -/
  theorem expressionNames_run : ∀ (available : List String) (expression : Expr),
      Run (.expressionNames available expression)
        (Decode.expressionNamesResolve available expression)
    | _, .literal _ => Derives.done _
    | _, .name _ => Derives.done _
    | available, .call _ args =>
        Derives.ask_done (body := body) (expressionsNames_run available args)
    | available, .memory address =>
        Derives.ask_done (body := body) (expressionNames_run available address)
    | available, .binary _ left right =>
        Derives.ask (body := body) (called := expressionNames_run available left)
          (Derives.ask (body := body) (called := expressionNames_run available right)
            (Derives.done _))
    | available, .conditional thenValue condition elseValue =>
        Derives.ask (body := body) (called := expressionNames_run available thenValue)
          (Derives.ask (body := body) (called := expressionNames_run available condition)
            (Derives.ask (body := body) (called := expressionNames_run available elseValue)
              (Derives.done _)))

  /-- Every expression-list name request has a checked derivation of the real validator. -/
  theorem expressionsNames_run : ∀ (available : List String) (expressions : List Expr),
      Run (.expressionsNames available expressions)
        (Decode.expressionsNamesResolve available expressions)
    | _, [] => Derives.done _
    | available, expression :: rest =>
        Derives.ask (body := body) (called := expressionNames_run available expression)
          (Derives.ask (body := body) (called := expressionsNames_run available rest)
            (Derives.done _))
end

/-- Every statement name request has a checked derivation of the real validator. -/
theorem statementsNames_run : ∀ (fuel : Nat) (available : List String) (statements : List Stmt),
    Run (.statementsNames fuel available statements)
      (Decode.statementsNamesResolve fuel available statements)
  | 0, _, _ => Derives.done _
  | _ + 1, _, [] => Derives.done _
  | fuel + 1, available, statement :: rest => by
      cases statement with
      | bind name value =>
          exact Derives.ask (body := body) (called := expressionNames_run available value)
            (Derives.ask (body := body) (called := statementsNames_run fuel (name :: available) rest)
              (Derives.done _))
      | store address value =>
          exact Derives.ask (body := body) (called := expressionNames_run available address)
            (Derives.ask (body := body) (called := expressionNames_run available value)
              (Derives.ask (body := body) (called := statementsNames_run fuel available rest)
                (Derives.done _)))
      | return_ value =>
          exact Derives.ask (body := body) (called := expressionNames_run available value)
            (Derives.ask (body := body) (called := statementsNames_run fuel available rest)
              (Derives.done _))
      | expression value =>
          exact Derives.ask (body := body) (called := expressionNames_run available value)
            (Derives.ask (body := body) (called := statementsNames_run fuel available rest)
              (Derives.done _))
      | if_ condition thenBody elseBody =>
          exact Derives.ask (body := body) (called := expressionNames_run available condition)
            (Derives.ask (body := body) (called := statementsNames_run fuel available thenBody)
              (match elseBody with
                | none =>
                    Derives.ask (body := body)
                      (called := statementsNames_run fuel available rest)
                      (Derives.done _)
                | some elseBody =>
                    Derives.ask (body := body)
                      (called := statementsNames_run fuel available elseBody)
                      (Derives.ask (body := body)
                        (called := statementsNames_run fuel available rest)
                        (Derives.done _))))

/-- Every local-name request is a checked leaf of the real validator. -/
theorem resolveNames_run (constants : List ConstantDecl) (functions : List FunctionDecl) :
    Run (.resolveNames constants functions) (Decode.resolveNames constants functions) :=
  Derives.done _

end Slang.Derivation.Validation
