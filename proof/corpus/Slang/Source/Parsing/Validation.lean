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

/-- A delimiter prefix represented as an implication composes with its tail derivation. -/
theorem «continue» {stack0 stack1 : List TokenKind} {tokens0 tokens1 : List Token}
    (prefixStep : DelimitersBalancedDerivation tokens1 stack1 →
      DelimitersBalancedDerivation tokens0 stack0)
    (tail : DelimitersBalancedDerivation tokens1 stack1) :
    DelimitersBalancedDerivation tokens0 stack0 := prefixStep tail

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

/-- A balanced chunk that empties the stack composes after a chunk that leaves the empty stack. -/
theorem append {tokens0 tokens1 : List Token} {stack : List TokenKind}
    (first : DelimitersBalancedDerivation tokens0 stack)
    (second : DelimitersBalancedDerivation tokens1 []) :
    DelimitersBalancedDerivation (tokens0 ++ tokens1) stack := by
  induction first with
  | done => exact second
  | lparen _ tailAppend => exact .lparen tailAppend
  | lbrace _ tailAppend => exact .lbrace tailAppend
  | lbracket _ tailAppend => exact .lbracket tailAppend
  | rparen _ tailAppend => exact .rparen tailAppend
  | rbrace _ tailAppend => exact .rbrace tailAppend
  | rbracket _ tailAppend => exact .rbracket tailAppend
  | other head _ tailAppend => exact .other head tailAppend

end DelimitersBalancedDerivation

/-- The two successful guards of `Decode.spansValid`. -/
inductive SpansValidDerivation : Nat → List Span → Prop where
  | intro {size : Nat} {spans : List Span}
      (bounds : spans.all (fun span => span.start < span.stop && span.stop ≤ size) = true)
      (separated : spans.all (fun first => spans.all fun second =>
        first == second || first.stop ≤ second.start || second.stop ≤ first.start) = true) :
      SpansValidDerivation size spans

namespace SpansValidDerivation

/-- Boolean universal guards compose over list append. -/
theorem all_append (predicate : Span → Bool) {a b : List Span} (first : a.all predicate = true)
    (second : b.all predicate = true) : (a ++ b).all predicate = true := by
  induction a with
  | nil => exact second
  | cons value rest inductionHypothesis =>
      unfold List.all at first ⊢
      change (predicate value && (rest ++ b).all predicate) = true
      cases head : predicate value with
      | false =>
          rw [head] at first
          cases first
      | true =>
          rw [head] at first
          exact inductionHypothesis first

/-- A successful nonempty universal guard includes its head guard. -/
theorem all_head (predicate : Span → Bool) {value : Span} {rest : List Span}
    (guard : (value :: rest).all predicate = true) : predicate value = true := by
  unfold List.all at guard
  cases head : predicate value with
  | false => rw [head] at guard; cases guard
  | true => rfl

/-- A successful nonempty universal guard includes its tail guard. -/
theorem all_tail (predicate : Span → Bool) {value : Span} {rest : List Span}
    (guard : (value :: rest).all predicate = true) : rest.all predicate = true := by
  unfold List.all at guard
  cases head : predicate value with
  | false => rw [head] at guard; cases guard
  | true => rw [head] at guard; exact guard

/-- Nested universal guards combine when their inner lists append. -/
theorem all_inner_append (relation : Span → Span → Bool) {outer a b : List Span}
    (first : outer.all (fun value => a.all (relation value)) = true)
    (second : outer.all (fun value => b.all (relation value)) = true) :
    outer.all (fun value => (a ++ b).all (relation value)) = true := by
  induction outer with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      have headCombined := all_append (relation value)
        (all_head (fun item => a.all (relation item)) first)
        (all_head (fun item => b.all (relation item)) second)
      have tailCombined := inductionHypothesis
        (all_tail (fun item => a.all (relation item)) first)
        (all_tail (fun item => b.all (relation item)) second)
      unfold List.all
      change ((a ++ b).all (relation value) &&
        rest.all (fun item => (a ++ b).all (relation item))) = true
      rw [headCombined, tailCombined]
      rfl

/-- Valid span chunks compose when both directed cross-chunk separation guards hold. -/
theorem append {size : Nat} {a b : List Span} (first : SpansValidDerivation size a)
    (second : SpansValidDerivation size b)
    (crossAB : a.all (fun first => b.all fun second =>
      first == second || first.stop ≤ second.start || second.stop ≤ first.start) = true)
    (crossBA : b.all (fun first => a.all fun second =>
      first == second || first.stop ≤ second.start || second.stop ≤ first.start) = true) :
    SpansValidDerivation size (a ++ b) := by
  cases first with
  | intro boundsA separatedA =>
      cases second with
      | intro boundsB separatedB =>
          apply SpansValidDerivation.intro
          · exact all_append _ boundsA boundsB
          · exact all_append _ (all_inner_append _ separatedA crossAB)
              (all_inner_append _ crossBA separatedB)

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

/-- Reversing into an accumulator preserves the combined list length. -/
theorem reverseAux_length (values accumulator : List String) :
    (List.reverseAux values accumulator).length = values.length + accumulator.length := by
  induction values generalizing accumulator with
  | nil =>
      unfold List.reverseAux
      exact (Nat.zero_add _).symm
  | cons value rest inductionHypothesis =>
      unfold List.reverseAux
      simpa only [List.length, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        inductionHypothesis (value :: accumulator)

/-- Absence from a string list makes its equality search fail. -/
theorem any_beq_false_of_not_mem {name : String} (values : List String) (absent : name ∉ values) :
    values.any (fun value => name == value) = false := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      unfold List.any
      change ((name == value) || rest.any fun item => name == item) = false
      cases equal : name == value with
      | false =>
          exact inductionHypothesis (fun membership => absent (.tail value membership))
      | true =>
          have same : name = value := LawfulBEq.eq_of_beq equal
          exact False.elim (absent (same ▸ .head rest))

/-- The deduplication loop cannot exceed its remaining-input and accumulator lengths. -/
theorem eraseDupsLoop_length_le (values accumulator : List String) :
    (List.eraseDupsBy.loop (fun left right : String => left == right) values accumulator).length ≤
      values.length + accumulator.length := by
  induction values generalizing accumulator with
  | nil =>
      unfold List.eraseDupsBy.loop List.reverse
      have reversed := reverseAux_length accumulator []
      exact Nat.le_of_eq (by simpa only [List.length, Nat.add_zero, Nat.zero_add] using reversed)
  | cons value rest inductionHypothesis =>
      unfold List.eraseDupsBy.loop
      split
      · exact Nat.le_trans (inductionHypothesis accumulator)
          (Nat.add_le_add_right (Nat.le.step (Nat.le_refl rest.length)) accumulator.length)
      · simpa only [List.length, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          inductionHypothesis (value :: accumulator)

/-- A duplicate-free remainder makes the deduplication loop retain every element. -/
theorem eraseDupsLoop_length_of_nodup {values accumulator : List String}
    (unique : values.Nodup) (cross : ∀ value, value ∈ values → value ∉ accumulator) :
    (List.eraseDupsBy.loop (fun left right : String => left == right) values accumulator).length =
      values.length + accumulator.length := by
  induction values generalizing accumulator with
  | nil =>
      unfold List.eraseDupsBy.loop List.reverse
      have reversed := reverseAux_length accumulator []
      simpa only [List.length, Nat.add_zero, Nat.zero_add] using reversed
  | cons value rest inductionHypothesis =>
      cases unique with
      | cons headUnique tailUnique =>
          have absent : value ∉ accumulator := cross value (.head rest)
          have anyFalse := any_beq_false_of_not_mem accumulator absent
          unfold List.eraseDupsBy.loop
          rw [anyFalse]
          have tailCross : ∀ item, item ∈ rest → item ∉ value :: accumulator := by
            intro item membership combinedMembership
            cases combinedMembership with
            | head => exact headUnique value membership rfl
            | tail _ accumulatorMembership =>
                exact cross item (.tail value membership) accumulatorMembership
          have := inductionHypothesis tailUnique tailCross
          simpa only [List.length, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using this

/-- Propositional no-duplicates entails the executable distinctness guard. -/
theorem of_nodup {names : List String} (guard : names.Nodup) : DistinctDerivation names := by
  apply DistinctDerivation.intro
  unfold List.eraseDups List.eraseDupsBy
  exact eraseDupsLoop_length_of_nodup guard
    (fun value membership accumulatorMembership => nomatch accumulatorMembership)

/-- Membership in an appended list originates in one of its two inputs. -/
theorem mem_append_cases {name : String} (a b : List String) : name ∈ a ++ b → name ∈ a ∨ name ∈ b := by
  induction a with
  | nil => exact fun membership => .inr membership
  | cons value rest inductionHypothesis =>
      intro membership
      cases membership with
      | head => exact .inl (.head rest)
      | tail _ tailMembership =>
          cases inductionHypothesis tailMembership with
          | inl inRest => exact .inl (.tail value inRest)
          | inr inB => exact .inr inB

/-- Duplicate-free string lists append under a directed disjointness hypothesis. -/
theorem nodup_append_of_cross {a b : List String} (first : a.Nodup) (second : b.Nodup)
    (cross : ∀ name, name ∈ a → name ∉ b) : (a ++ b).Nodup := by
  induction a with
  | nil => exact second
  | cons value rest inductionHypothesis =>
      cases first with
      | cons headUnique tailUnique =>
          apply List.Pairwise.cons
          · intro item membership equal
            cases mem_append_cases rest b membership with
            | inl inRest => exact headUnique item inRest equal
            | inr inB => exact cross value (.head rest) (equal ▸ inB)
          · exact inductionHypothesis tailUnique
              (fun item membership => cross item (.tail value membership))

/-- Distinct name chunks compose when no name in the first occurs in the second. -/
theorem append {a b : List String} (_first : DistinctDerivation a)
    (_second : DistinctDerivation b) (firstNodup : a.Nodup) (secondNodup : b.Nodup)
    (cross : ∀ name ∈ a, name ∉ b) :
    DistinctDerivation (a ++ b) := by
  exact of_nodup (nodup_append_of_cross firstNodup secondNodup cross)

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

namespace ExpressionsCallsResolveDerivation

/-- Expression-list call derivations compose under one global available-name list. -/
theorem append {available : List String} {a b : List Expr}
    (first : ExpressionsCallsResolveDerivation available a)
    (second : ExpressionsCallsResolveDerivation available b) :
    ExpressionsCallsResolveDerivation available (a ++ b) := by
  induction a with
  | nil => exact second
  | cons expression rest tailAppend =>
      cases first with
      | next head tail => exact .next head (tailAppend tail)

end ExpressionsCallsResolveDerivation

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

namespace StatementsCallsResolveDerivation

/-- Statement-list call derivations compose under one global available-name list. -/
theorem append {available : List String} {a b : List Stmt}
    (first : StatementsCallsResolveDerivation available a)
    (second : StatementsCallsResolveDerivation available b) :
    StatementsCallsResolveDerivation available (a ++ b) := by
  induction a with
  | nil => exact second
  | cons statement rest tailAppend =>
      cases first with
      | next head tail => exact .next head (tailAppend tail)

end StatementsCallsResolveDerivation

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

/-- Function-list call derivations compose under one global available-name list. -/
theorem append {available : List String} {a b : List FunctionDecl}
    (first : FunctionCallsResolveDerivation available a)
    (second : FunctionCallsResolveDerivation available b) :
    FunctionCallsResolveDerivation available (a ++ b) := by
  induction a with
  | nil => exact second
  | cons function rest tailAppend =>
      cases first with
      | next body tail => exact .next body (tailAppend tail)

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

/-- Globally checked function chunks assemble when both use the full combined name list. -/
theorem of_parts {a b : List FunctionDecl}
    (first : FunctionCallsResolveDerivation ((a ++ b).map (·.name)) a)
    (second : FunctionCallsResolveDerivation ((a ++ b).map (·.name)) b) :
    CallsResolveDerivation (a ++ b) := .intro (first.append second)

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


namespace ExpressionsNamesResolveDerivation

/-- Expression-list name derivations compose under one available-name list. -/
theorem append {available : List String} {a b : List Expr}
    (first : ExpressionsNamesResolveDerivation available a)
    (second : ExpressionsNamesResolveDerivation available b) :
    ExpressionsNamesResolveDerivation available (a ++ b) := by
  induction a with
  | nil => exact second
  | cons expression rest tailAppend =>
      cases first with
      | next head tail => exact .next head (tailAppend tail)

end ExpressionsNamesResolveDerivation

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

/-- A statement-name prefix represented as an implication composes with its tail derivation. -/
theorem «continue» {fuel0 fuel1 : Nat} {available0 available1 : List String}
    {statements0 statements1 : List Stmt}
    (prefixStep : StatementsNamesResolveDerivation fuel1 available1 statements1 →
      StatementsNamesResolveDerivation fuel0 available0 statements0)
    (tail : StatementsNamesResolveDerivation fuel1 available1 statements1) :
    StatementsNamesResolveDerivation fuel0 available0 statements0 := prefixStep tail

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

/-- Function-list name derivations compose under one constant-name list. -/
theorem append {constantNames : List String} {a b : List FunctionDecl}
    (first : FunctionNamesResolveDerivation constantNames a)
    (second : FunctionNamesResolveDerivation constantNames b) :
    FunctionNamesResolveDerivation constantNames (a ++ b) := by
  induction a with
  | nil => exact second
  | cons function rest tailAppend =>
      cases first with
      | next parameters body tail => exact .next parameters body (tailAppend tail)

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

/-- Complete name-resolution derivations compose for function chunks sharing all constants. -/
theorem append {constants : List ConstantDecl} {a b : List FunctionDecl}
    (first : NamesResolveDerivation constants a) (second : NamesResolveDerivation constants b) :
    NamesResolveDerivation constants (a ++ b) := by
  cases first with
  | intro firstFunctions =>
      cases second with
      | intro secondFunctions => exact .intro (firstFunctions.append secondFunctions)

/-- Complete structural name success entails `Decode.resolveNames = true`. -/
theorem sound {constants : List ConstantDecl} {functions : List FunctionDecl}
    (derivation : NamesResolveDerivation constants functions) :
    Decode.resolveNames constants functions = true := by
  cases derivation with
  | intro functionsDerivation =>
      unfold Decode.resolveNames
      exact functionsDerivation.sound

end NamesResolveDerivation

namespace DistinctDerivation

/-- Executable distinctness entails the structural distinctness derivation. -/
theorem of_bool {names : List String} (guard : Decode.distinct names = true) :
    DistinctDerivation names := by
  apply DistinctDerivation.intro
  unfold Decode.distinct at guard
  exact of_decide_eq_true guard

end DistinctDerivation

/-- Left projection of a successful `Bool.and`. -/
private theorem andL {left right : Bool} (guard : (left && right) = true) : left = true := by
  cases left with
  | false => cases guard
  | true => rfl

/-- Right projection of a successful `Bool.and`. -/
private theorem andR {left right : Bool} (guard : (left && right) = true) : right = true := by
  cases left with
  | false => cases guard
  | true => exact guard

mutual
  /-- Executable call resolution of an expression yields its structural derivation. -/
  theorem ExpressionCallsResolveDerivation.of_bool {available : List String} {expression : Expr}
      (guard : Decode.expressionCallsResolve available expression = true) :
      ExpressionCallsResolveDerivation available expression := by
    cases expression with
    | literal value => exact .literal value
    | name name => exact .name name
    | call callee args =>
        change (available.contains callee &&
          Decode.expressionsCallsResolve available args) = true at guard
        exact .call (andL guard) (ExpressionsCallsResolveDerivation.of_bool (andR guard))
    | memory address =>
        change Decode.expressionCallsResolve available address = true at guard
        exact .memory (ExpressionCallsResolveDerivation.of_bool guard)
    | binary op left right =>
        change (Decode.expressionCallsResolve available left &&
          Decode.expressionCallsResolve available right) = true at guard
        exact .binary (ExpressionCallsResolveDerivation.of_bool (andL guard))
          (ExpressionCallsResolveDerivation.of_bool (andR guard))
    | conditional thenValue condition elseValue =>
        change (Decode.expressionCallsResolve available thenValue &&
          Decode.expressionCallsResolve available condition &&
          Decode.expressionCallsResolve available elseValue) = true at guard
        exact .conditional (ExpressionCallsResolveDerivation.of_bool (andL (andL guard)))
          (ExpressionCallsResolveDerivation.of_bool (andR (andL guard)))
          (ExpressionCallsResolveDerivation.of_bool (andR guard))

  /-- Executable call resolution of an expression list yields its structural derivation. -/
  theorem ExpressionsCallsResolveDerivation.of_bool {available : List String}
      {expressions : List Expr}
      (guard : Decode.expressionsCallsResolve available expressions = true) :
      ExpressionsCallsResolveDerivation available expressions := by
    cases expressions with
    | nil => exact .done
    | cons expression rest =>
        change (Decode.expressionCallsResolve available expression &&
          Decode.expressionsCallsResolve available rest) = true at guard
        exact .next (ExpressionCallsResolveDerivation.of_bool (andL guard))
          (ExpressionsCallsResolveDerivation.of_bool (andR guard))

  /-- Executable call resolution of a statement yields its structural derivation. -/
  theorem StatementCallsResolveDerivation.of_bool {available : List String} {statement : Stmt}
      (guard : Decode.statementCallsResolve available statement = true) :
      StatementCallsResolveDerivation available statement := by
    cases statement with
    | bind name value =>
        change Decode.expressionCallsResolve available value = true at guard
        exact .bind (ExpressionCallsResolveDerivation.of_bool guard)
    | store address value =>
        change (Decode.expressionCallsResolve available address &&
          Decode.expressionCallsResolve available value) = true at guard
        exact .store (ExpressionCallsResolveDerivation.of_bool (andL guard))
          (ExpressionCallsResolveDerivation.of_bool (andR guard))
    | return_ value =>
        change Decode.expressionCallsResolve available value = true at guard
        exact .return_ (ExpressionCallsResolveDerivation.of_bool guard)
    | expression value =>
        change Decode.expressionCallsResolve available value = true at guard
        exact .expression (ExpressionCallsResolveDerivation.of_bool guard)
    | if_ condition thenBody elseBody =>
        cases elseBody with
        | none =>
            change (Decode.expressionCallsResolve available condition &&
              Decode.statementsCallsResolve available thenBody && true) = true at guard
            exact .ifNone (ExpressionCallsResolveDerivation.of_bool (andL (andL guard)))
              (StatementsCallsResolveDerivation.of_bool (andR (andL guard)))
        | some elseBody =>
            change (Decode.expressionCallsResolve available condition &&
              Decode.statementsCallsResolve available thenBody &&
              Decode.statementsCallsResolve available elseBody) = true at guard
            exact .ifSome (ExpressionCallsResolveDerivation.of_bool (andL (andL guard)))
              (StatementsCallsResolveDerivation.of_bool (andR (andL guard)))
              (StatementsCallsResolveDerivation.of_bool (andR guard))

  /-- Executable call resolution of a statement list yields its structural derivation. -/
  theorem StatementsCallsResolveDerivation.of_bool {available : List String}
      {statements : List Stmt}
      (guard : Decode.statementsCallsResolve available statements = true) :
      StatementsCallsResolveDerivation available statements := by
    cases statements with
    | nil => exact .done
    | cons statement rest =>
        change (Decode.statementCallsResolve available statement &&
          Decode.statementsCallsResolve available rest) = true at guard
        exact .next (StatementCallsResolveDerivation.of_bool (andL guard))
          (StatementsCallsResolveDerivation.of_bool (andR guard))
end

/-- Executable per-function call resolution yields the structural function-list derivation. -/
theorem FunctionCallsResolveDerivation.of_bool {available : List String}
    {functions : List FunctionDecl}
    (guard : functions.all (fun function =>
      Decode.statementsCallsResolve available function.body) = true) :
    FunctionCallsResolveDerivation available functions := by
  induction functions with
  | nil => exact .done
  | cons function rest inductionHypothesis =>
      rw [List.all_cons] at guard
      exact .next (StatementsCallsResolveDerivation.of_bool (andL guard))
        (inductionHypothesis (andR guard))

/-- `Decode.resolveCalls = true` yields the complete structural call derivation. -/
theorem CallsResolveDerivation.of_bool {functions : List FunctionDecl}
    (guard : Decode.resolveCalls functions = true) : CallsResolveDerivation functions := by
  change (functions.all fun function =>
    Decode.statementsCallsResolve (functions.map (·.name)) function.body) = true at guard
  exact .intro (FunctionCallsResolveDerivation.of_bool guard)

mutual
  /-- Executable name resolution of an expression yields its structural derivation. -/
  theorem ExpressionNamesResolveDerivation.of_bool {available : List String} {expression : Expr}
      (guard : Decode.expressionNamesResolve available expression = true) :
      ExpressionNamesResolveDerivation available expression := by
    cases expression with
    | literal value => exact .literal value
    | name name =>
        change available.contains name = true at guard
        exact .name guard
    | call callee args =>
        change Decode.expressionsNamesResolve available args = true at guard
        exact .call (ExpressionsNamesResolveDerivation.of_bool guard)
    | memory address =>
        change Decode.expressionNamesResolve available address = true at guard
        exact .memory (ExpressionNamesResolveDerivation.of_bool guard)
    | binary op left right =>
        change (Decode.expressionNamesResolve available left &&
          Decode.expressionNamesResolve available right) = true at guard
        exact .binary (ExpressionNamesResolveDerivation.of_bool (andL guard))
          (ExpressionNamesResolveDerivation.of_bool (andR guard))
    | conditional thenValue condition elseValue =>
        change (Decode.expressionNamesResolve available thenValue &&
          Decode.expressionNamesResolve available condition &&
          Decode.expressionNamesResolve available elseValue) = true at guard
        exact .conditional (ExpressionNamesResolveDerivation.of_bool (andL (andL guard)))
          (ExpressionNamesResolveDerivation.of_bool (andR (andL guard)))
          (ExpressionNamesResolveDerivation.of_bool (andR guard))

  /-- Executable name resolution of an expression list yields its structural derivation. -/
  theorem ExpressionsNamesResolveDerivation.of_bool {available : List String}
      {expressions : List Expr}
      (guard : Decode.expressionsNamesResolve available expressions = true) :
      ExpressionsNamesResolveDerivation available expressions := by
    cases expressions with
    | nil => exact .done
    | cons expression rest =>
        change (Decode.expressionNamesResolve available expression &&
          Decode.expressionsNamesResolve available rest) = true at guard
        exact .next (ExpressionNamesResolveDerivation.of_bool (andL guard))
          (ExpressionsNamesResolveDerivation.of_bool (andR guard))
end

/-- Executable fuel-bounded statement name resolution yields its structural derivation. -/
theorem StatementsNamesResolveDerivation.of_bool :
    ∀ (fuel : Nat) (available : List String) (statements : List Stmt),
      Decode.statementsNamesResolve fuel available statements = true →
      StatementsNamesResolveDerivation fuel available statements := by
  intro fuel
  induction fuel with
  | zero =>
      intro available statements guard
      change false = true at guard
      cases guard
  | succ predecessor inductionHypothesis =>
      intro available statements guard
      cases statements with
      | nil => exact .done
      | cons statement rest =>
          cases statement with
          | bind name value =>
              change (Decode.expressionNamesResolve available value &&
                Decode.statementsNamesResolve predecessor (name :: available) rest) = true
                at guard
              exact .bind (ExpressionNamesResolveDerivation.of_bool (andL guard))
                (inductionHypothesis (name :: available) rest (andR guard))
          | store address value =>
              change (Decode.expressionNamesResolve available address &&
                Decode.expressionNamesResolve available value &&
                Decode.statementsNamesResolve predecessor available rest) = true at guard
              exact .store (ExpressionNamesResolveDerivation.of_bool (andL (andL guard)))
                (ExpressionNamesResolveDerivation.of_bool (andR (andL guard)))
                (inductionHypothesis available rest (andR guard))
          | return_ value =>
              change (Decode.expressionNamesResolve available value &&
                Decode.statementsNamesResolve predecessor available rest) = true at guard
              exact .return_ (ExpressionNamesResolveDerivation.of_bool (andL guard))
                (inductionHypothesis available rest (andR guard))
          | expression value =>
              change (Decode.expressionNamesResolve available value &&
                Decode.statementsNamesResolve predecessor available rest) = true at guard
              exact .expression (ExpressionNamesResolveDerivation.of_bool (andL guard))
                (inductionHypothesis available rest (andR guard))
          | if_ condition thenBody elseBody =>
              cases elseBody with
              | none =>
                  change (Decode.expressionNamesResolve available condition &&
                    Decode.statementsNamesResolve predecessor available thenBody && true &&
                    Decode.statementsNamesResolve predecessor available rest) = true at guard
                  exact .ifNone
                    (ExpressionNamesResolveDerivation.of_bool (andL (andL (andL guard))))
                    (inductionHypothesis available thenBody (andR (andL (andL guard))))
                    (inductionHypothesis available rest (andR guard))
              | some elseBody =>
                  change (Decode.expressionNamesResolve available condition &&
                    Decode.statementsNamesResolve predecessor available thenBody &&
                    Decode.statementsNamesResolve predecessor available elseBody &&
                    Decode.statementsNamesResolve predecessor available rest) = true at guard
                  exact .ifSome
                    (ExpressionNamesResolveDerivation.of_bool (andL (andL (andL guard))))
                    (inductionHypothesis available thenBody (andR (andL (andL guard))))
                    (inductionHypothesis available elseBody (andR (andL guard)))
                    (inductionHypothesis available rest (andR guard))

/-- Executable per-function name resolution yields the structural function-list derivation. -/
theorem FunctionNamesResolveDerivation.of_bool {constantNames : List String}
    {functions : List FunctionDecl}
    (guard : functions.all (fun function => Decode.distinct function.parameters &&
      Decode.statementsNamesResolve (function.body.length * 4 + 64)
        (function.parameters ++ constantNames) function.body) = true) :
    FunctionNamesResolveDerivation constantNames functions := by
  induction functions with
  | nil => exact .done
  | cons function rest inductionHypothesis =>
      rw [List.all_cons] at guard
      exact .next (DistinctDerivation.of_bool (andL (andL guard)))
        (StatementsNamesResolveDerivation.of_bool (function.body.length * 4 + 64)
          (function.parameters ++ constantNames) function.body (andR (andL guard)))
        (inductionHypothesis (andR guard))

/-- `Decode.resolveNames = true` yields the complete structural name derivation. -/
theorem NamesResolveDerivation.of_bool {constants : List ConstantDecl}
    {functions : List FunctionDecl} (guard : Decode.resolveNames constants functions = true) :
    NamesResolveDerivation constants functions := by
  change (functions.all fun function => Decode.distinct function.parameters &&
    Decode.statementsNamesResolve (function.body.length * 4 + 64)
      (function.parameters ++ constants.map (·.name)) function.body) = true at guard
  exact .intro (FunctionNamesResolveDerivation.of_bool guard)

end Slang.Source.Parsing
