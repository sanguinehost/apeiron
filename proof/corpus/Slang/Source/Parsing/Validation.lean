import Slang.Source.Parsing.Statement

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Parsing

open Slang Slang.Decode

/-- A token that leaves the delimiter stack unchanged. -/
inductive NonDelimiterToken : Token → Prop where
  | ident (name : String) (span : Span) : NonDelimiterToken { kind := .ident name, span }
  | numeral (value : Nat) (span : Span) : NonDelimiterToken { kind := .numeral value, span }
  | comma (span : Span) : NonDelimiterToken { kind := .comma, span }
  | semi (span : Span) : NonDelimiterToken { kind := .semi, span }
  | newline (span : Span) : NonDelimiterToken { kind := .newline, span }
  | assign (span : Span) : NonDelimiterToken { kind := .assign, span }
  | lt (span : Span) : NonDelimiterToken { kind := .lt, span }
  | le (span : Span) : NonDelimiterToken { kind := .le, span }
  | gt (span : Span) : NonDelimiterToken { kind := .gt, span }
  | ge (span : Span) : NonDelimiterToken { kind := .ge, span }
  | other (value : Char) (span : Span) : NonDelimiterToken { kind := .other value, span }

/-- A successful structural traversal of `Decode.delimitersBalanced`. -/
inductive DelimitersBalancedDerivation : List Token → List TokenKind → Prop where
  | done : DelimitersBalancedDerivation [] []
  | lparen {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest (.lparen :: stack)) :
      DelimitersBalancedDerivation ({ kind := .lparen, span } :: rest) stack
  | lbrace {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest (.lbrace :: stack)) :
      DelimitersBalancedDerivation ({ kind := .lbrace, span } :: rest) stack
  | lbracket {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest (.lbracket :: stack)) :
      DelimitersBalancedDerivation ({ kind := .lbracket, span } :: rest) stack
  | rparen {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest stack) :
      DelimitersBalancedDerivation ({ kind := .rparen, span } :: rest) (.lparen :: stack)
  | rbrace {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest stack) :
      DelimitersBalancedDerivation ({ kind := .rbrace, span } :: rest) (.lbrace :: stack)
  | rbracket {span : Span} {rest : List Token} {stack : List TokenKind}
      (tail : DelimitersBalancedDerivation rest stack) :
      DelimitersBalancedDerivation ({ kind := .rbracket, span } :: rest) (.lbracket :: stack)
  | other {token : Token} {rest : List Token} {stack : List TokenKind}
      (head : NonDelimiterToken token)
      (tail : DelimitersBalancedDerivation rest stack) :
      DelimitersBalancedDerivation (token :: rest) stack

namespace DelimitersBalancedDerivation

/-- A successful delimiter traversal entails acceptance by the executable validator. -/
theorem sound {tokens : List Token} {stack : List TokenKind}
    (derivation : DelimitersBalancedDerivation tokens stack) :
    Decode.delimitersBalanced tokens stack = true := by
  induction derivation with
  | done => rfl
  | lparen _ tailSound => exact tailSound
  | lbrace _ tailSound => exact tailSound
  | lbracket _ tailSound => exact tailSound
  | rparen _ tailSound => exact tailSound
  | rbrace _ tailSound => exact tailSound
  | rbracket _ tailSound => exact tailSound
  | other head _ tailSound =>
      cases head <;> exact tailSound

end DelimitersBalancedDerivation

/-- The two successful guards of `Decode.spansValid`. -/
inductive SpansValidDerivation : Nat → List Span → Prop where
  | intro {size : Nat} {spans : List Span}
      (bounds : spans.all (fun span => span.start < span.stop && span.stop ≤ size) = true)
      (separated : spans.all (fun first => spans.all fun second =>
        first == second || first.stop ≤ second.start || second.stop ≤ first.start) = true) :
      SpansValidDerivation size spans

namespace SpansValidDerivation

/-- Successful span guards entail acceptance by the executable validator. -/
theorem sound {size : Nat} {spans : List Span} (derivation : SpansValidDerivation size spans) :
    Decode.spansValid size spans = true := by
  cases derivation with
  | intro bounds separated =>
      unfold Decode.spansValid
      rw [bounds, separated]
      rfl

end SpansValidDerivation

/-- The successful equality guard of `Decode.distinct`. -/
inductive DistinctDerivation : List String → Prop where
  | intro {names : List String} (guard : names.eraseDups.length = names.length) :
      DistinctDerivation names

namespace DistinctDerivation

/-- A successful length guard entails acceptance by the executable validator. -/
theorem sound {names : List String} (derivation : DistinctDerivation names) :
    Decode.distinct names = true := by
  cases derivation with
  | intro guard =>
      unfold Decode.distinct
      exact decide_eq_true guard

end DistinctDerivation

mutual
  /-- Structural success for one expression in call resolution. -/
  inductive ExpressionCallsResolveDerivation (available : List String) : Expr → Prop where
    | literal (value : Nat) : ExpressionCallsResolveDerivation available (.literal value)
    | name (name : String) : ExpressionCallsResolveDerivation available (.name name)
    | call {callee : String} {args : List Expr} (calleeGuard : available.contains callee = true)
        (argsDerivation : ExpressionsCallsResolveDerivation available args) :
        ExpressionCallsResolveDerivation available (.call callee args)
    | memory {address : Expr} (addressDerivation : ExpressionCallsResolveDerivation available address) :
        ExpressionCallsResolveDerivation available (.memory address)
    | binary {op : BinOp} {left right : Expr}
        (leftDerivation : ExpressionCallsResolveDerivation available left)
        (rightDerivation : ExpressionCallsResolveDerivation available right) :
        ExpressionCallsResolveDerivation available (.binary op left right)
    | conditional {thenValue condition elseValue : Expr}
        (thenDerivation : ExpressionCallsResolveDerivation available thenValue)
        (conditionDerivation : ExpressionCallsResolveDerivation available condition)
        (elseDerivation : ExpressionCallsResolveDerivation available elseValue) :
        ExpressionCallsResolveDerivation available (.conditional thenValue condition elseValue)

  /-- Structural success for an expression list in call resolution. -/
  inductive ExpressionsCallsResolveDerivation (available : List String) : List Expr → Prop where
    | done : ExpressionsCallsResolveDerivation available []
    | next {expression : Expr} {rest : List Expr}
        (head : ExpressionCallsResolveDerivation available expression)
        (tail : ExpressionsCallsResolveDerivation available rest) :
        ExpressionsCallsResolveDerivation available (expression :: rest)
end

mutual
  /-- Structural expression call success entails the executable result. -/
  theorem ExpressionCallsResolveDerivation.sound {available : List String} {expression : Expr}
      (derivation : ExpressionCallsResolveDerivation available expression) :
      Decode.expressionCallsResolve available expression = true := by
    cases derivation with
    | literal _ => rfl
    | name _ => rfl
    | call calleeGuard argsDerivation =>
        rw [Decode.expressionCallsResolve, calleeGuard, argsDerivation.sound]
        rfl
    | memory addressDerivation => exact addressDerivation.sound
    | binary leftDerivation rightDerivation =>
        rw [Decode.expressionCallsResolve, leftDerivation.sound, rightDerivation.sound]
        rfl
    | conditional thenDerivation conditionDerivation elseDerivation =>
        rw [Decode.expressionCallsResolve, thenDerivation.sound, conditionDerivation.sound,
          elseDerivation.sound]
        rfl

  /-- Structural expression-list call success entails the executable result. -/
  theorem ExpressionsCallsResolveDerivation.sound {available : List String}
      {expressions : List Expr} (derivation : ExpressionsCallsResolveDerivation available expressions) :
      Decode.expressionsCallsResolve available expressions = true := by
    cases derivation with
    | done => rfl
    | next head tail =>
        rw [Decode.expressionsCallsResolve, head.sound, tail.sound]
        rfl
end

mutual
  /-- Structural success for one statement in call resolution. -/
  inductive StatementCallsResolveDerivation (available : List String) : Stmt → Prop where
    | bind {name : String} {value : Expr}
        (valueDerivation : ExpressionCallsResolveDerivation available value) :
        StatementCallsResolveDerivation available (.bind name value)
    | store {address value : Expr}
        (addressDerivation : ExpressionCallsResolveDerivation available address)
        (valueDerivation : ExpressionCallsResolveDerivation available value) :
        StatementCallsResolveDerivation available (.store address value)
    | return_ {value : Expr} (valueDerivation : ExpressionCallsResolveDerivation available value) :
        StatementCallsResolveDerivation available (.return_ value)
    | expression {value : Expr}
        (valueDerivation : ExpressionCallsResolveDerivation available value) :
        StatementCallsResolveDerivation available (.expression value)
    | ifNone {condition : Expr} {thenBody : List Stmt}
        (conditionDerivation : ExpressionCallsResolveDerivation available condition)
        (thenDerivation : StatementsCallsResolveDerivation available thenBody) :
        StatementCallsResolveDerivation available (.if_ condition thenBody none)
    | ifSome {condition : Expr} {thenBody elseBody : List Stmt}
        (conditionDerivation : ExpressionCallsResolveDerivation available condition)
        (thenDerivation : StatementsCallsResolveDerivation available thenBody)
        (elseDerivation : StatementsCallsResolveDerivation available elseBody) :
        StatementCallsResolveDerivation available (.if_ condition thenBody (some elseBody))

  /-- Structural success for a statement list in call resolution. -/
  inductive StatementsCallsResolveDerivation (available : List String) : List Stmt → Prop where
    | done : StatementsCallsResolveDerivation available []
    | next {statement : Stmt} {rest : List Stmt}
        (head : StatementCallsResolveDerivation available statement)
        (tail : StatementsCallsResolveDerivation available rest) :
        StatementsCallsResolveDerivation available (statement :: rest)
end

mutual
  /-- Structural statement call success entails the executable result. -/
  theorem StatementCallsResolveDerivation.sound {available : List String} {statement : Stmt}
      (derivation : StatementCallsResolveDerivation available statement) :
      Decode.statementCallsResolve available statement = true := by
    cases derivation with
    | bind valueDerivation => exact valueDerivation.sound
    | store addressDerivation valueDerivation =>
        rw [Decode.statementCallsResolve, addressDerivation.sound, valueDerivation.sound]
        rfl
    | return_ valueDerivation => exact valueDerivation.sound
    | expression valueDerivation => exact valueDerivation.sound
    | ifNone conditionDerivation thenDerivation =>
        rw [Decode.statementCallsResolve, conditionDerivation.sound, thenDerivation.sound]
        rfl
    | ifSome conditionDerivation thenDerivation elseDerivation =>
        rw [Decode.statementCallsResolve, conditionDerivation.sound, thenDerivation.sound,
          elseDerivation.sound]
        rfl

  /-- Structural statement-list call success entails the executable result. -/
  theorem StatementsCallsResolveDerivation.sound {available : List String} {statements : List Stmt}
      (derivation : StatementsCallsResolveDerivation available statements) :
      Decode.statementsCallsResolve available statements = true := by
    cases derivation with
    | done => rfl
    | next head tail =>
        rw [Decode.statementsCallsResolve, head.sound, tail.sound]
        rfl
end

/-- Structural success for every function body under one available-name list. -/
inductive FunctionCallsResolveDerivation (available : List String) : List FunctionDecl → Prop where
  | done : FunctionCallsResolveDerivation available []
  | next {function : FunctionDecl} {rest : List FunctionDecl}
      (body : StatementsCallsResolveDerivation available function.body)
      (tail : FunctionCallsResolveDerivation available rest) :
      FunctionCallsResolveDerivation available (function :: rest)

namespace FunctionCallsResolveDerivation

/-- Structural function-list call success entails the `List.all` result. -/
theorem sound {available : List String} {functions : List FunctionDecl}
    (derivation : FunctionCallsResolveDerivation available functions) :
    functions.all (fun function => Decode.statementsCallsResolve available function.body) = true := by
  induction derivation with
  | done => rfl
  | next body _ tailSound =>
      rw [List.all_cons, body.sound, tailSound]
      rfl

end FunctionCallsResolveDerivation

/-- Structural success for the complete reachable-call pass. -/
inductive CallsResolveDerivation : List FunctionDecl → Prop where
  | intro {functions : List FunctionDecl}
      (functionsDerivation : FunctionCallsResolveDerivation (functions.map (·.name)) functions) :
      CallsResolveDerivation functions

namespace CallsResolveDerivation

/-- Complete structural call success entails `Decode.resolveCalls = true`. -/
theorem sound {functions : List FunctionDecl} (derivation : CallsResolveDerivation functions) :
    Decode.resolveCalls functions = true := by
  cases derivation with
  | intro functionsDerivation =>
      unfold Decode.resolveCalls
      exact functionsDerivation.sound

end CallsResolveDerivation

mutual
  /-- Structural success for one expression in local-name resolution. -/
  inductive ExpressionNamesResolveDerivation (available : List String) : Expr → Prop where
    | literal (value : Nat) : ExpressionNamesResolveDerivation available (.literal value)
    | name {name : String} (guard : available.contains name = true) :
        ExpressionNamesResolveDerivation available (.name name)
    | call {callee : String} {args : List Expr}
        (argsDerivation : ExpressionsNamesResolveDerivation available args) :
        ExpressionNamesResolveDerivation available (.call callee args)
    | memory {address : Expr} (addressDerivation : ExpressionNamesResolveDerivation available address) :
        ExpressionNamesResolveDerivation available (.memory address)
    | binary {op : BinOp} {left right : Expr}
        (leftDerivation : ExpressionNamesResolveDerivation available left)
        (rightDerivation : ExpressionNamesResolveDerivation available right) :
        ExpressionNamesResolveDerivation available (.binary op left right)
    | conditional {thenValue condition elseValue : Expr}
        (thenDerivation : ExpressionNamesResolveDerivation available thenValue)
        (conditionDerivation : ExpressionNamesResolveDerivation available condition)
        (elseDerivation : ExpressionNamesResolveDerivation available elseValue) :
        ExpressionNamesResolveDerivation available (.conditional thenValue condition elseValue)

  /-- Structural success for an expression list in local-name resolution. -/
  inductive ExpressionsNamesResolveDerivation (available : List String) : List Expr → Prop where
    | done : ExpressionsNamesResolveDerivation available []
    | next {expression : Expr} {rest : List Expr}
        (head : ExpressionNamesResolveDerivation available expression)
        (tail : ExpressionsNamesResolveDerivation available rest) :
        ExpressionsNamesResolveDerivation available (expression :: rest)
end

mutual
  /-- Structural expression-name success entails the executable result. -/
  theorem ExpressionNamesResolveDerivation.sound {available : List String} {expression : Expr}
      (derivation : ExpressionNamesResolveDerivation available expression) :
      Decode.expressionNamesResolve available expression = true := by
    cases derivation with
    | literal _ => rfl
    | name guard => exact guard
    | call argsDerivation => exact argsDerivation.sound
    | memory addressDerivation => exact addressDerivation.sound
    | binary leftDerivation rightDerivation =>
        rw [Decode.expressionNamesResolve, leftDerivation.sound, rightDerivation.sound]
        rfl
    | conditional thenDerivation conditionDerivation elseDerivation =>
        rw [Decode.expressionNamesResolve, thenDerivation.sound, conditionDerivation.sound,
          elseDerivation.sound]
        rfl

  /-- Structural expression-list name success entails the executable result. -/
  theorem ExpressionsNamesResolveDerivation.sound {available : List String}
      {expressions : List Expr} (derivation : ExpressionsNamesResolveDerivation available expressions) :
      Decode.expressionsNamesResolve available expressions = true := by
    cases derivation with
    | done => rfl
    | next head tail =>
        rw [Decode.expressionsNamesResolve, head.sound, tail.sound]
        rfl
end

/-- Structural success for the fuel-bounded statement name pass. -/
inductive StatementsNamesResolveDerivation : Nat → List String → List Stmt → Prop where
  | done {fuel : Nat} {available : List String} :
      StatementsNamesResolveDerivation (fuel + 1) available []
  | bind {fuel : Nat} {available : List String} {name : String} {value : Expr} {rest : List Stmt}
      (valueDerivation : ExpressionNamesResolveDerivation available value)
      (tail : StatementsNamesResolveDerivation fuel (name :: available) rest) :
      StatementsNamesResolveDerivation (fuel + 1) available (.bind name value :: rest)
  | store {fuel : Nat} {available : List String} {address value : Expr} {rest : List Stmt}
      (addressDerivation : ExpressionNamesResolveDerivation available address)
      (valueDerivation : ExpressionNamesResolveDerivation available value)
      (tail : StatementsNamesResolveDerivation fuel available rest) :
      StatementsNamesResolveDerivation (fuel + 1) available (.store address value :: rest)
  | return_ {fuel : Nat} {available : List String} {value : Expr} {rest : List Stmt}
      (valueDerivation : ExpressionNamesResolveDerivation available value)
      (tail : StatementsNamesResolveDerivation fuel available rest) :
      StatementsNamesResolveDerivation (fuel + 1) available (.return_ value :: rest)
  | expression {fuel : Nat} {available : List String} {value : Expr} {rest : List Stmt}
      (valueDerivation : ExpressionNamesResolveDerivation available value)
      (tail : StatementsNamesResolveDerivation fuel available rest) :
      StatementsNamesResolveDerivation (fuel + 1) available (.expression value :: rest)
  | ifNone {fuel : Nat} {available : List String} {condition : Expr}
      {thenBody rest : List Stmt}
      (conditionDerivation : ExpressionNamesResolveDerivation available condition)
      (thenDerivation : StatementsNamesResolveDerivation fuel available thenBody)
      (tail : StatementsNamesResolveDerivation fuel available rest) :
      StatementsNamesResolveDerivation (fuel + 1) available
        (.if_ condition thenBody none :: rest)
  | ifSome {fuel : Nat} {available : List String} {condition : Expr}
      {thenBody elseBody rest : List Stmt}
      (conditionDerivation : ExpressionNamesResolveDerivation available condition)
      (thenDerivation : StatementsNamesResolveDerivation fuel available thenBody)
      (elseDerivation : StatementsNamesResolveDerivation fuel available elseBody)
      (tail : StatementsNamesResolveDerivation fuel available rest) :
      StatementsNamesResolveDerivation (fuel + 1) available
        (.if_ condition thenBody (some elseBody) :: rest)

namespace StatementsNamesResolveDerivation

/-- Structural statement-name success entails the executable fuel-bounded result. -/
theorem sound {fuel : Nat} {available : List String} {statements : List Stmt}
    (derivation : StatementsNamesResolveDerivation fuel available statements) :
    Decode.statementsNamesResolve fuel available statements = true := by
  cases derivation with
  | done => rfl
  | bind valueDerivation tail =>
      rw [Decode.statementsNamesResolve, valueDerivation.sound, tail.sound]
      rfl
  | store addressDerivation valueDerivation tail =>
      rw [Decode.statementsNamesResolve, addressDerivation.sound, valueDerivation.sound, tail.sound]
      rfl
  | return_ valueDerivation tail =>
      rw [Decode.statementsNamesResolve, valueDerivation.sound, tail.sound]
      rfl
  | expression valueDerivation tail =>
      rw [Decode.statementsNamesResolve, valueDerivation.sound, tail.sound]
      rfl
  | ifNone conditionDerivation thenDerivation tail =>
      rw [Decode.statementsNamesResolve, conditionDerivation.sound, thenDerivation.sound, tail.sound]
      rfl
  | ifSome conditionDerivation thenDerivation elseDerivation tail =>
      rw [Decode.statementsNamesResolve, conditionDerivation.sound, thenDerivation.sound,
        elseDerivation.sound, tail.sound]
      rfl

end StatementsNamesResolveDerivation

/-- Structural local-name success for every function. -/
inductive FunctionNamesResolveDerivation (constantNames : List String) : List FunctionDecl → Prop where
  | done : FunctionNamesResolveDerivation constantNames []
  | next {function : FunctionDecl} {rest : List FunctionDecl}
      (parameters : DistinctDerivation function.parameters)
      (body : StatementsNamesResolveDerivation (function.body.length * 4 + 64)
        (function.parameters ++ constantNames) function.body)
      (tail : FunctionNamesResolveDerivation constantNames rest) :
      FunctionNamesResolveDerivation constantNames (function :: rest)

namespace FunctionNamesResolveDerivation

/-- Structural function-list name success entails the `List.all` result. -/
theorem sound {constantNames : List String} {functions : List FunctionDecl}
    (derivation : FunctionNamesResolveDerivation constantNames functions) :
    functions.all (fun function => Decode.distinct function.parameters &&
      Decode.statementsNamesResolve (function.body.length * 4 + 64)
        (function.parameters ++ constantNames) function.body) = true := by
  induction derivation with
  | done => rfl
  | next parameters body _ tailSound =>
      rw [List.all_cons, parameters.sound, body.sound, tailSound]
      rfl

end FunctionNamesResolveDerivation

/-- Structural success for the complete local-name pass. -/
inductive NamesResolveDerivation : List ConstantDecl → List FunctionDecl → Prop where
  | intro {constants : List ConstantDecl} {functions : List FunctionDecl}
      (functionsDerivation : FunctionNamesResolveDerivation (constants.map (·.name)) functions) :
      NamesResolveDerivation constants functions

namespace NamesResolveDerivation

/-- Complete structural name success entails `Decode.resolveNames = true`. -/
theorem sound {constants : List ConstantDecl} {functions : List FunctionDecl}
    (derivation : NamesResolveDerivation constants functions) :
    Decode.resolveNames constants functions = true := by
  cases derivation with
  | intro functionsDerivation =>
      unfold Decode.resolveNames
      exact functionsDerivation.sound

end NamesResolveDerivation

end Slang.Source.Parsing
