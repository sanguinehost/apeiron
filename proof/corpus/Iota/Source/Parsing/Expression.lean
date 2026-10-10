import Iota.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.Parsing

open Iota Iota.Decode

/-- The `is not` lookahead does not override the ordinary binary-parser branch. -/
inductive OrdinaryBinaryGuard : Token → List Token → Prop where
  | notIs {token : Token} {rest : List Token} (guard : token.kind ≠ .ident "is") :
      OrdinaryBinaryGuard token rest
  | isEnd (span : Span) : OrdinaryBinaryGuard { kind := .ident "is", span } []
  | isNext {span : Span} {next : Token} {tail : List Token}
      (guard : next.kind ≠ .ident "not") :
      OrdinaryBinaryGuard { kind := .ident "is", span } (next :: tail)

mutual
  /-- A successful structural traversal of `Decode.parseExprFuel`. -/
  inductive ExprScan : Nat → Nat → List Token → Expr → List Token → Prop where
    | step {fuel minPrec : Nat} {tokens primaryRest leftover : List Token}
        {primary expression : Expr}
        (primaryScan : PrimaryScan fuel tokens primary primaryRest)
        (binaryScan : BinaryScan fuel minPrec primary primaryRest expression leftover) :
        ExprScan (fuel + 1) minPrec tokens expression leftover

  /-- A successful structural traversal of `Decode.parsePrimaryFuel`. -/
  inductive PrimaryScan : Nat → List Token → Expr → List Token → Prop where
    | numeral {fuel value : Nat} {tokens rest : List Token} {span : Span}
        (drop : Decode.dropNewlines tokens = { kind := .numeral value, span } :: rest) :
        PrimaryScan (fuel + 1) tokens (.literal value) rest
    | nameEnd {fuel : Nat} {tokens : List Token} {name : String} {span : Span}
        (drop : Decode.dropNewlines tokens = [{ kind := .ident name, span }])
        (notMemory : name ≠ "memory") :
        PrimaryScan (fuel + 1) tokens (.name name) []
    | nameNext {fuel : Nat} {tokens rest : List Token} {name : String} {span : Span}
        {next : Token}
        (drop : Decode.dropNewlines tokens = { kind := .ident name, span } :: next :: rest)
        (notMemory : name ≠ "memory") (notCall : next.kind ≠ .lparen) :
        PrimaryScan (fuel + 1) tokens (.name name) (next :: rest)
    | call {fuel : Nat} {tokens tail leftover : List Token} {name : String}
        {nameSpan openSpan : Span} {args : List Expr}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident name, span := nameSpan } :: { kind := .lparen, span := openSpan } :: tail)
        (notMemory : name ≠ "memory") (argsScan : ArgsScan fuel tail args leftover) :
        PrimaryScan (fuel + 1) tokens (.call name args) leftover
    | memory {fuel : Nat} {tokens afterOpen afterAddress leftover : List Token}
        {nameSpan openSpan closeSpan : Span} {address : Expr}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "memory", span := nameSpan } ::
            { kind := .lbracket, span := openSpan } :: afterOpen)
        (addressScan : ExprScan fuel 0 afterOpen address afterAddress)
        (close : afterAddress = { kind := .rbracket, span := closeSpan } :: leftover) :
        PrimaryScan (fuel + 1) tokens (.memory address) leftover
    | parenthesized {fuel : Nat} {tokens afterOpen afterValue leftover : List Token}
        {openSpan closeSpan : Span} {value : Expr}
        (drop : Decode.dropNewlines tokens = { kind := .lparen, span := openSpan } :: afterOpen)
        (valueScan : ExprScan fuel 0 afterOpen value afterValue)
        (close : afterValue = { kind := .rparen, span := closeSpan } :: leftover) :
        PrimaryScan (fuel + 1) tokens value leftover

  /-- A successful structural traversal of `Decode.parseArgsFuel`. -/
  inductive ArgsScan : Nat → List Token → List Expr → List Token → Prop where
    | done {fuel : Nat} {tokens rest : List Token} {span : Span}
        (drop : Decode.dropNewlines tokens = { kind := .rparen, span } :: rest) :
        ArgsScan (fuel + 1) tokens [] rest
    | comma {fuel : Nat} {tokens expressionInput afterArgument nextTail leftover : List Token}
        {token next : Token} {argument : Expr} {arguments : List Expr}
        (drop : Decode.dropNewlines tokens = token :: expressionInput)
        (notClose : token.kind ≠ .rparen)
        (argumentScan : ExprScan fuel 0 (token :: expressionInput) argument afterArgument)
        (afterDrop : Decode.dropNewlines afterArgument = next :: nextTail)
        (comma : next.kind = .comma)
        (tailScan : ArgsScan fuel nextTail arguments leftover) :
        ArgsScan (fuel + 1) tokens (argument :: arguments) leftover
    | last {fuel : Nat} {tokens expressionInput afterArgument nextTail : List Token}
        {token next : Token} {argument : Expr}
        (drop : Decode.dropNewlines tokens = token :: expressionInput)
        (notClose : token.kind ≠ .rparen)
        (argumentScan : ExprScan fuel 0 (token :: expressionInput) argument afterArgument)
        (afterDrop : Decode.dropNewlines afterArgument = next :: nextTail)
        (notComma : next.kind ≠ .comma) (close : next.kind = .rparen) :
        ArgsScan (fuel + 1) tokens [argument] nextTail

  /-- A successful structural traversal of `Decode.parseBinaryFuel`. -/
  inductive BinaryScan : Nat → Nat → Expr → List Token → Expr → List Token → Prop where
    | done {fuel minPrec : Nat} {left : Expr} {tokens : List Token}
        (drop : Decode.dropNewlines tokens = []) :
        BinaryScan (fuel + 1) minPrec left tokens left []
    | belowPrec {fuel minPrec prec : Nat} {left : Expr} {tokens rest : List Token}
        {token : Token} {op : BinOp}
        (drop : Decode.dropNewlines tokens = token :: rest)
        (ordinary : OrdinaryBinaryGuard token rest)
        (operator : Decode.binOp token.kind = some (op, prec)) (below : prec < minPrec) :
        BinaryScan (fuel + 1) minPrec left tokens left tokens
    | operatorStep {fuel minPrec prec : Nat} {left right expression : Expr}
        {tokens rest afterRight leftover : List Token} {token : Token} {op : BinOp}
        (drop : Decode.dropNewlines tokens = token :: rest)
        (ordinary : OrdinaryBinaryGuard token rest)
        (operator : Decode.binOp token.kind = some (op, prec)) (notBelow : ¬ prec < minPrec)
        (rightScan : ExprScan fuel (prec + 1) rest right afterRight)
        (tailScan : BinaryScan fuel minPrec (.binary op left right) afterRight expression leftover) :
        BinaryScan (fuel + 1) minPrec left tokens expression leftover
    | conditionalStop {fuel minPrec : Nat} {left : Expr} {tokens rest : List Token}
        {span : Span}
        (drop : Decode.dropNewlines tokens = { kind := .ident "if", span } :: rest)
        (notIs : (TokenKind.ident "if") ≠ .ident "is")
        (notOperator : Decode.binOp (.ident "if") = none) (nested : 0 < minPrec) :
        BinaryScan (fuel + 1) minPrec left tokens left tokens
    | conditional {fuel minPrec : Nat} {left condition elseValue : Expr}
        {tokens rest afterCondition elseInput leftover : List Token} {ifSpan elseSpan : Span}
        (drop : Decode.dropNewlines tokens = { kind := .ident "if", span := ifSpan } :: rest)
        (notIs : (TokenKind.ident "if") ≠ .ident "is")
        (notOperator : Decode.binOp (.ident "if") = none) (notNested : ¬ 0 < minPrec)
        (conditionScan : ExprScan fuel 0 rest condition afterCondition)
        (afterDrop : Decode.dropNewlines afterCondition =
          { kind := .ident "else", span := elseSpan } :: elseInput)
        (elseScan : ExprScan fuel 0 elseInput elseValue leftover) :
        BinaryScan (fuel + 1) minPrec left tokens
          (.conditional left condition elseValue) leftover
    | stop {fuel minPrec : Nat} {left : Expr} {tokens rest : List Token} {token : Token}
        (drop : Decode.dropNewlines tokens = token :: rest)
        (ordinary : OrdinaryBinaryGuard token rest)
        (notOperator : Decode.binOp token.kind = none) (notConditional : token.kind ≠ .ident "if") :
        BinaryScan (fuel + 1) minPrec left tokens left tokens
    | isNotStop {fuel minPrec : Nat} {left : Expr} {tokens tail : List Token}
        {isSpan notSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "is", span := isSpan } :: { kind := .ident "not", span := notSpan } :: tail)
        (nested : 3 < minPrec) :
        BinaryScan (fuel + 1) minPrec left tokens left tokens
    | isNot {fuel minPrec : Nat} {left right expression : Expr}
        {tokens tail afterRight leftover : List Token} {isSpan notSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "is", span := isSpan } :: { kind := .ident "not", span := notSpan } :: tail)
        (notNested : ¬ 3 < minPrec) (rightScan : ExprScan fuel 4 tail right afterRight)
        (tailScan : BinaryScan fuel minPrec (.binary .ne left right) afterRight expression leftover) :
        BinaryScan (fuel + 1) minPrec left tokens expression leftover
end

mutual
  /-- A structural expression scan entails the executable expression-parser result. -/
  theorem ExprScan.sound {fuel minPrec : Nat} {tokens leftover : List Token} {expression : Expr}
      (scan : ExprScan fuel minPrec tokens expression leftover) :
      Decode.parseExprFuel fuel minPrec tokens = .ok (expression, leftover) := by
    cases scan with
    | step primaryScan binaryScan =>
        unfold Decode.parseExprFuel
        rw [primaryScan.sound]
        exact binaryScan.sound

  /-- A structural primary scan entails the executable primary-parser result. -/
  theorem PrimaryScan.sound {fuel : Nat} {tokens leftover : List Token} {expression : Expr}
      (scan : PrimaryScan fuel tokens expression leftover) :
      Decode.parsePrimaryFuel fuel tokens = .ok (expression, leftover) := by
    cases scan with
    | numeral drop =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
    | nameEnd drop notMemory =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
        dsimp only
        rw [if_neg notMemory]
    | nameNext drop notMemory notCall =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
        dsimp only
        rw [if_neg notMemory, if_neg notCall]
    | call drop notMemory argsScan =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
        dsimp only
        rw [if_neg notMemory, if_pos rfl, argsScan.sound]
        rfl
    | memory drop addressScan close =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
        dsimp only
        rw [if_pos rfl]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [addressScan.sound]
        change (Decode.expectKind .rbracket _).bind _ = _
        rw [close]
        unfold Decode.expectKind
        dsimp only
        rw [if_pos rfl]
        rfl
    | parenthesized drop valueScan close =>
        unfold Decode.parsePrimaryFuel
        rw [drop]
        dsimp only
        rw [valueScan.sound, close]
        rfl

  /-- A structural argument scan entails the executable argument-parser result. -/
  theorem ArgsScan.sound {fuel : Nat} {tokens leftover : List Token} {arguments : List Expr}
      (scan : ArgsScan fuel tokens arguments leftover) :
      Decode.parseArgsFuel fuel tokens = .ok (arguments, leftover) := by
    cases scan with
    | done drop =>
        unfold Decode.parseArgsFuel
        rw [drop]
        dsimp only
        rw [if_pos rfl]
    | comma drop notClose argumentScan afterDrop comma tailScan =>
        unfold Decode.parseArgsFuel
        rw [drop]
        dsimp only
        rw [if_neg notClose, argumentScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail =>
              if next.kind = TokenKind.comma then
                (Decode.parseArgsFuel _ tail).bind fun result => Except.ok (_ :: result.1, result.2)
              else if next.kind = TokenKind.rparen then Except.ok ([_], tail)
              else Except.error "expected comma or closing parenthesis"
          | [] => Except.error "unterminated call arguments") = _
        rw [afterDrop]
        dsimp only
        rw [if_pos comma, tailScan.sound]
        rfl
    | last drop notClose argumentScan afterDrop notComma close =>
        unfold Decode.parseArgsFuel
        rw [drop]
        dsimp only
        rw [if_neg notClose, argumentScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail =>
              if next.kind = TokenKind.comma then
                (Decode.parseArgsFuel _ tail).bind fun result => Except.ok (_ :: result.1, result.2)
              else if next.kind = TokenKind.rparen then Except.ok ([_], tail)
              else Except.error "expected comma or closing parenthesis"
          | [] => Except.error "unterminated call arguments") = _
        rw [afterDrop]
        dsimp only
        rw [if_neg notComma, if_pos close]

  /-- A structural binary scan entails the executable precedence-parser result. -/
  theorem BinaryScan.sound {fuel minPrec : Nat} {left expression : Expr}
      {tokens leftover : List Token} (scan : BinaryScan fuel minPrec left tokens expression leftover) :
      Decode.parseBinaryFuel fuel minPrec left tokens = .ok (expression, leftover) := by
    cases scan with
    | done drop =>
        unfold Decode.parseBinaryFuel
        rw [drop]
    | belowPrec drop ordinary operator below =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        cases ordinary with
        | notIs notIs =>
            rw [if_neg notIs, operator]
            dsimp only
            rw [if_pos below]
        | isEnd =>
            rw [if_pos rfl]
            dsimp only
            rw [operator]
            dsimp only
            rw [if_pos below]
        | isNext notNot =>
            rw [if_pos rfl]
            dsimp only
            rw [if_neg notNot, operator]
            dsimp only
            rw [if_pos below]
    | operatorStep drop ordinary operator notBelow rightScan tailScan =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        cases ordinary with
        | notIs notIs =>
            rw [if_neg notIs, operator]
            dsimp only
            rw [if_neg notBelow, rightScan.sound]
            exact tailScan.sound
        | isEnd =>
            rw [if_pos rfl]
            dsimp only
            rw [operator]
            dsimp only
            rw [if_neg notBelow, rightScan.sound]
            exact tailScan.sound
        | isNext notNot =>
            rw [if_pos rfl]
            dsimp only
            rw [if_neg notNot, operator]
            dsimp only
            rw [if_neg notBelow, rightScan.sound]
            exact tailScan.sound
    | conditionalStop drop notIs notOperator nested =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        rw [if_neg notIs, notOperator, if_pos rfl, if_pos nested]
    | conditional drop notIs notOperator notNested conditionScan afterDrop elseScan =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        rw [if_neg notIs, notOperator, if_pos rfl, if_neg notNested, conditionScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail =>
              if next.kind = TokenKind.ident "else" then
                (Decode.parseExprFuel _ 0 tail).bind fun result =>
                  Except.ok (Expr.conditional _ _ result.1, result.2)
              else Except.error "expected inline else"
          | [] => Except.error "unterminated inline conditional") = _
        rw [afterDrop]
        dsimp only
        rw [if_pos rfl, elseScan.sound]
        rfl
    | stop drop ordinary notOperator notConditional =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        cases ordinary with
        | notIs notIs => rw [if_neg notIs, notOperator, if_neg notConditional]
        | isEnd =>
            rw [if_pos rfl]
            dsimp only
            rw [notOperator, if_neg notConditional]
        | isNext notNot =>
            rw [if_pos rfl]
            dsimp only
            rw [if_neg notNot, notOperator, if_neg notConditional]
    | isNotStop drop nested =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        rw [if_pos rfl, if_pos rfl, if_pos nested]
    | isNot drop notNested rightScan tailScan =>
        unfold Decode.parseBinaryFuel
        rw [drop]
        dsimp only
        rw [if_pos rfl, if_pos rfl, if_neg notNested, rightScan.sound]
        exact tailScan.sound
end

/-- A top-level expression scan uses the executable wrapper's fuel and precedence. -/
inductive ExprScanTop : List Token → Expr → List Token → Prop where
  | intro {tokens leftover : List Token} {expression : Expr}
      (scan : ExprScan (tokens.length * 4 + 16) 0 tokens expression leftover) :
      ExprScanTop tokens expression leftover

namespace ExprScanTop

/-- A top-level structural scan entails the expression parser result used by declarations. -/
theorem sound {tokens leftover : List Token} {expression : Expr}
    (scan : ExprScanTop tokens expression leftover) :
    Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, leftover) := by
  cases scan with
  | intro expressionScan => exact expressionScan.sound

end ExprScanTop

end Iota.Source.Parsing
