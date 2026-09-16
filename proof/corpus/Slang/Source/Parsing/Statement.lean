import Slang.Source.Parsing.Expression

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Parsing

open Slang Slang.Decode

/-- A successful structural traversal of `Decode.parseParameters`. -/
inductive ParameterScan : Nat → List Token → List String → List Token → Prop where
  | done {fuel : Nat} {span : Span} {rest : List Token} :
      ParameterScan (fuel + 1) ({ kind := .rparen, span } :: rest) [] rest
  | comma {fuel : Nat} {name : String} {nameSpan commaSpan : Span}
      {tail leftover : List Token} {names : List String}
      (scan : ParameterScan fuel tail names leftover) :
      ParameterScan (fuel + 1)
        ({ kind := .ident name, span := nameSpan } :: { kind := .comma, span := commaSpan } :: tail)
        (name :: names) leftover
  | last {fuel : Nat} {name : String} {nameSpan closeSpan : Span} {leftover : List Token} :
      ParameterScan (fuel + 1)
        ({ kind := .ident name, span := nameSpan } :: { kind := .rparen, span := closeSpan } :: leftover)
        [name] leftover

namespace ParameterScan

/-- A structural parameter scan entails the executable parameter-parser result. -/
theorem sound {fuel : Nat} {tokens leftover : List Token} {names : List String}
    (scan : ParameterScan fuel tokens names leftover) :
    Decode.parseParameters fuel tokens = .ok (names, leftover) := by
  induction scan with
  | done =>
      unfold Decode.parseParameters
      rw [if_pos rfl]
  | comma _ tailSound =>
      unfold Decode.parseParameters
      dsimp only
      rw [tailSound]
      rfl
  | last =>
      unfold Decode.parseParameters
      dsimp only
      rfl

end ParameterScan

/-- A token entering the grouped expression-statement branch of `Decode.parseStmtFuel`. -/
inductive ExpressionStatementToken : Token → Prop where
  | numeral (value : Nat) (span : Span) : ExpressionStatementToken { kind := .numeral value, span }
  | lparen (span : Span) : ExpressionStatementToken { kind := .lparen, span }
  | rparen (span : Span) : ExpressionStatementToken { kind := .rparen, span }
  | lbrace (span : Span) : ExpressionStatementToken { kind := .lbrace, span }
  | rbrace (span : Span) : ExpressionStatementToken { kind := .rbrace, span }
  | lbracket (span : Span) : ExpressionStatementToken { kind := .lbracket, span }
  | rbracket (span : Span) : ExpressionStatementToken { kind := .rbracket, span }
  | comma (span : Span) : ExpressionStatementToken { kind := .comma, span }
  | semi (span : Span) : ExpressionStatementToken { kind := .semi, span }
  | newline (span : Span) : ExpressionStatementToken { kind := .newline, span }
  | assign (span : Span) : ExpressionStatementToken { kind := .assign, span }
  | lt (span : Span) : ExpressionStatementToken { kind := .lt, span }
  | le (span : Span) : ExpressionStatementToken { kind := .le, span }
  | gt (span : Span) : ExpressionStatementToken { kind := .gt, span }
  | ge (span : Span) : ExpressionStatementToken { kind := .ge, span }
  | other (value : Char) (span : Span) : ExpressionStatementToken { kind := .other value, span }

mutual
  /-- A successful structural traversal of `Decode.parseStmtFuel`. -/
  inductive StmtScan : Nat → List Token → Stmt → List Token → Prop where
    | return_ {fuel : Nat} {tokens input afterValue leftover : List Token} {value : Expr}
        {keywordSpan semiSpan : Span}
        (drop : Decode.dropNewlines tokens = { kind := .ident "return", span := keywordSpan } :: input)
        (valueScan : ExprScanTop input value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.return_ value) leftover
    | ifNoElseEnd {fuel : Nat} {tokens conditionInput afterCondition bodyInput afterThen : List Token}
        {condition : Expr} {thenBody : List Stmt} {keywordSpan openSpan closeSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "if", span := keywordSpan } :: conditionInput)
        (conditionScan : ExprScanTop conditionInput condition afterCondition)
        (openBrace : Decode.dropNewlines afterCondition =
          { kind := .lbrace, span := openSpan } :: bodyInput)
        (thenScan : BlockScan fuel bodyInput thenBody { kind := .rbrace, span := closeSpan } afterThen)
        (afterDrop : Decode.dropNewlines afterThen = []) :
        StmtScan (fuel + 1) tokens (.if_ condition thenBody none) afterThen
    | ifNoElseNext {fuel : Nat} {tokens conditionInput afterCondition bodyInput afterThen tail : List Token}
        {condition : Expr} {thenBody : List Stmt} {next : Token}
        {keywordSpan openSpan closeSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "if", span := keywordSpan } :: conditionInput)
        (conditionScan : ExprScanTop conditionInput condition afterCondition)
        (openBrace : Decode.dropNewlines afterCondition =
          { kind := .lbrace, span := openSpan } :: bodyInput)
        (thenScan : BlockScan fuel bodyInput thenBody { kind := .rbrace, span := closeSpan } afterThen)
        (afterDrop : Decode.dropNewlines afterThen = next :: tail)
        (notElse : next.kind ≠ .ident "else") :
        StmtScan (fuel + 1) tokens (.if_ condition thenBody none) afterThen
    | ifElse {fuel : Nat}
        {tokens conditionInput afterCondition thenInput afterThen elseOpenInput elseInput leftover : List Token}
        {condition : Expr} {thenBody elseBody : List Stmt}
        {keywordSpan thenOpenSpan thenCloseSpan elseSpan elseOpenSpan elseCloseSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "if", span := keywordSpan } :: conditionInput)
        (conditionScan : ExprScanTop conditionInput condition afterCondition)
        (thenOpen : Decode.dropNewlines afterCondition =
          { kind := .lbrace, span := thenOpenSpan } :: thenInput)
        (thenScan : BlockScan fuel thenInput thenBody
          { kind := .rbrace, span := thenCloseSpan } afterThen)
        (elseToken : Decode.dropNewlines afterThen =
          { kind := .ident "else", span := elseSpan } :: elseOpenInput)
        (elseOpen : Decode.dropNewlines elseOpenInput =
          { kind := .lbrace, span := elseOpenSpan } :: elseInput)
        (elseScan : BlockScan fuel elseInput elseBody
          { kind := .rbrace, span := elseCloseSpan } leftover) :
        StmtScan (fuel + 1) tokens (.if_ condition thenBody (some elseBody)) leftover
    | store {fuel : Nat} {tokens addressInput afterAddress afterClose valueInput afterValue leftover : List Token}
        {address value : Expr} {keywordSpan openSpan closeSpan assignSpan semiSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "memory", span := keywordSpan } ::
            { kind := .lbracket, span := openSpan } :: addressInput)
        (addressScan : ExprScanTop addressInput address afterAddress)
        (close : afterAddress = { kind := .rbracket, span := closeSpan } :: afterClose)
        (assign : Decode.dropNewlines afterClose =
          { kind := .assign, span := assignSpan } :: valueInput)
        (valueScan : ExprScanTop valueInput value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.store address value) leftover
    | memoryExpression {fuel : Nat} {tokens tail afterValue leftover : List Token}
        {value : Expr} {next : Token} {keywordSpan semiSpan : Span}
        (drop : Decode.dropNewlines tokens =
          { kind := .ident "memory", span := keywordSpan } :: next :: tail)
        (notBracket : next.kind ≠ .lbracket)
        (valueScan : ExprScanTop
          ({ kind := .ident "memory", span := keywordSpan } :: next :: tail) value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.expression value) leftover
    | bind {fuel : Nat} {tokens valueInput afterValue leftover : List Token}
        {name : String} {value : Expr} {nameSpan assignSpan semiSpan : Span}
        (notReturn : name ≠ "return") (notIf : name ≠ "if") (notMemory : name ≠ "memory")
        (drop : Decode.dropNewlines tokens =
          { kind := .ident name, span := nameSpan } ::
            { kind := .assign, span := assignSpan } :: valueInput)
        (valueScan : ExprScanTop valueInput value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.bind name value) leftover
    | nameExpression {fuel : Nat} {tokens tail afterValue leftover : List Token}
        {name : String} {value : Expr} {next : Token} {nameSpan semiSpan : Span}
        (notReturn : name ≠ "return") (notIf : name ≠ "if") (notMemory : name ≠ "memory")
        (drop : Decode.dropNewlines tokens =
          { kind := .ident name, span := nameSpan } :: next :: tail)
        (notAssign : next.kind ≠ .assign)
        (valueScan : ExprScanTop
          ({ kind := .ident name, span := nameSpan } :: next :: tail) value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.expression value) leftover
    | otherExpression {fuel : Nat} {tokens rest afterValue leftover : List Token}
        {token : Token} {value : Expr} {semiSpan : Span}
        (head : ExpressionStatementToken token)
        (drop : Decode.dropNewlines tokens = token :: rest)
        (valueScan : ExprScanTop (token :: rest) value afterValue)
        (semi : Decode.dropNewlines afterValue = { kind := .semi, span := semiSpan } :: leftover) :
        StmtScan (fuel + 1) tokens (.expression value) leftover

  /-- A successful structural traversal of `Decode.parseBlockFuel`. -/
  inductive BlockScan : Nat → List Token → List Stmt → Token → List Token → Prop where
    | done {fuel : Nat} {tokens leftover : List Token} {close : Token}
        (drop : Decode.dropNewlines tokens = close :: leftover)
        (closeKind : close.kind = .rbrace) :
        BlockScan (fuel + 1) tokens [] close leftover
    | next {fuel : Nat} {tokens statementInput afterStatement leftover : List Token}
        {token close : Token} {statement : Stmt} {statements : List Stmt}
        (drop : Decode.dropNewlines tokens = token :: statementInput)
        (notClose : token.kind ≠ .rbrace)
        (statementScan : StmtScan fuel (token :: statementInput) statement afterStatement)
        (tailScan : BlockScan fuel afterStatement statements close leftover) :
        BlockScan (fuel + 1) tokens (statement :: statements) close leftover
end

namespace BlockScan

/-- A structural block scan entails the executable block-parser result. -/
theorem sound {fuel : Nat} {tokens leftover : List Token} {statements : List Stmt}
    {terminator : Token} (scan : BlockScan fuel tokens statements terminator leftover) :
    Decode.parseBlockFuel fuel tokens = .ok (statements, terminator, leftover) := by
  induction scan using BlockScan.rec
      (motive_1 := fun fuel tokens statement leftover _ =>
        Decode.parseStmtFuel fuel tokens = .ok (statement, leftover)) with
  | return_ drop valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [valueScan.sound]
      change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
      rw [semi]
      rfl
  | ifNoElseEnd drop conditionScan openBrace _ afterDrop thenSound =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg (by decide), if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [conditionScan.sound]
      change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
      rw [openBrace]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [thenSound]
      change (match Decode.dropNewlines _ with | _ :: _ => _ | [] => _) = _
      rw [afterDrop]
  | ifNoElseNext drop conditionScan openBrace _ afterDrop notElse thenSound =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg (by decide), if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [conditionScan.sound]
      change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
      rw [openBrace]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [thenSound]
      change (match Decode.dropNewlines _ with | _ :: _ => _ | [] => _) = _
      rw [afterDrop]
      dsimp only
      rw [if_neg notElse]
  | ifElse drop conditionScan thenOpen _ elseToken elseOpen _ thenSound elseSound =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg (by decide), if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [conditionScan.sound]
      change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
      rw [thenOpen]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [thenSound]
      change (match Decode.dropNewlines _ with | _ :: _ => _ | [] => _) = _
      rw [elseToken]
      dsimp only
      rw [if_pos rfl, elseOpen]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [elseSound]
      rfl
  | store drop addressScan close assign valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg (by decide), if_neg (by decide), if_pos rfl, if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [addressScan.sound]
      change (Decode.expectKind .rbracket _).bind _ = _
      rw [close]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.expectKind .assign (Decode.dropNewlines _)).bind _ = _
      rw [assign]
      unfold Decode.expectKind
      dsimp only
      rw [if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [valueScan.sound]
      change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
      rw [semi]
      rfl
  | memoryExpression drop notBracket valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg (by decide), if_neg (by decide), if_pos rfl, if_neg notBracket]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [valueScan.sound]
      change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
      rw [semi]
      rfl
  | bind notReturn notIf notMemory drop valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg notReturn, if_neg notIf, if_neg notMemory, if_pos rfl]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [valueScan.sound]
      change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
      rw [semi]
      rfl
  | nameExpression notReturn notIf notMemory drop notAssign valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      dsimp only
      rw [if_neg notReturn, if_neg notIf, if_neg notMemory, if_neg notAssign]
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [valueScan.sound]
      change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
      rw [semi]
      rfl
  | otherExpression head drop valueScan semi =>
      unfold Decode.parseStmtFuel
      rw [drop]
      cases head <;> dsimp only <;>
        change (Decode.parseExprFuel _ 0 _).bind _ = _ <;>
        rw [valueScan.sound] <;>
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _ <;>
        rw [semi] <;> rfl
  | done drop closeKind =>
      unfold Decode.parseBlockFuel
      rw [drop]
      dsimp only
      rw [if_pos closeKind]
  | next drop notClose _ _ statementSound tailSound =>
      unfold Decode.parseBlockFuel
      rw [drop]
      dsimp only
      rw [if_neg notClose, statementSound]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [tailSound]
      rfl

end BlockScan

namespace StmtScan

  /-- A structural statement scan entails the executable statement-parser result. -/
  theorem sound {fuel : Nat} {tokens leftover : List Token} {statement : Stmt}
      (scan : StmtScan fuel tokens statement leftover) :
      Decode.parseStmtFuel fuel tokens = .ok (statement, leftover) := by
    cases scan with
    | return_ drop valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [valueScan.sound]
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
        rw [semi]
        rfl
    | ifNoElseEnd drop conditionScan openBrace thenScan afterDrop =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg (by decide), if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [conditionScan.sound]
        change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
        rw [openBrace]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.parseBlockFuel _ _).bind _ = _
        rw [thenScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail => if next.kind = TokenKind.ident "else" then _ else _
          | [] => _) = _
        rw [afterDrop]
    | ifNoElseNext drop conditionScan openBrace thenScan afterDrop notElse =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg (by decide), if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [conditionScan.sound]
        change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
        rw [openBrace]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.parseBlockFuel _ _).bind _ = _
        rw [thenScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail => if next.kind = TokenKind.ident "else" then _ else _
          | [] => _) = _
        rw [afterDrop]
        dsimp only
        rw [if_neg notElse]
    | ifElse drop conditionScan thenOpen thenScan elseToken elseOpen elseScan =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg (by decide), if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [conditionScan.sound]
        change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
        rw [thenOpen]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.parseBlockFuel _ _).bind _ = _
        rw [thenScan.sound]
        change (match Decode.dropNewlines _ with
          | next :: tail => if next.kind = TokenKind.ident "else" then _ else _
          | [] => _) = _
        rw [elseToken]
        dsimp only
        rw [if_pos rfl, elseOpen]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.parseBlockFuel _ _).bind _ = _
        rw [elseScan.sound]
        rfl
    | store drop addressScan close assign valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg (by decide), if_neg (by decide), if_pos rfl, if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [addressScan.sound]
        change (Decode.expectKind .rbracket _).bind _ = _
        rw [close]
        dsimp only [Decode.expectKind]
        rw [if_pos rfl]
        change (Decode.expectKind .assign (Decode.dropNewlines _)).bind _ = _
        rw [assign]
        unfold Decode.expectKind
        dsimp only
        rw [if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [valueScan.sound]
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
        rw [semi]
        rfl
    | memoryExpression drop notBracket valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg (by decide), if_neg (by decide), if_pos rfl, if_neg notBracket]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [valueScan.sound]
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
        rw [semi]
        rfl
    | bind notReturn notIf notMemory drop valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg notReturn, if_neg notIf, if_neg notMemory, if_pos rfl]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [valueScan.sound]
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
        rw [semi]
        rfl
    | nameExpression notReturn notIf notMemory drop notAssign valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        dsimp only
        rw [if_neg notReturn, if_neg notIf, if_neg notMemory, if_neg notAssign]
        change (Decode.parseExprFuel _ 0 _).bind _ = _
        rw [valueScan.sound]
        change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _
        rw [semi]
        rfl
    | otherExpression head drop valueScan semi =>
        unfold Decode.parseStmtFuel
        rw [drop]
        cases head <;> dsimp only <;>
          change (Decode.parseExprFuel _ 0 _).bind _ = _ <;>
          rw [valueScan.sound] <;>
          change (Decode.expectKind .semi (Decode.dropNewlines _)).bind _ = _ <;>
          rw [semi] <;> rfl

end StmtScan

/-- A successful structural traversal of `Decode.parseFunction`. -/
inductive FunctionScan : String → Nat → List Token → FunctionDecl → List Token → Prop where
  | intro {name : String} {start : Nat} {tokens parameterInput afterParameters bodyInput leftover : List Token}
      {parameters : List String} {body : List Stmt} {openParenSpan openBraceSpan : Span}
      {close : Token}
      (openParen : tokens = { kind := .lparen, span := openParenSpan } :: parameterInput)
      (parametersScan : ParameterScan (parameterInput.length + 1) parameterInput parameters afterParameters)
      (openBrace : Decode.dropNewlines afterParameters =
        { kind := .lbrace, span := openBraceSpan } :: bodyInput)
      (bodyScan : BlockScan (bodyInput.length + 16) bodyInput body close leftover) :
      FunctionScan name start tokens
        { name, parameters, body, span := ⟨start, close.span.stop⟩ } leftover

namespace FunctionScan

/-- A structural function scan entails the executable function-parser result. -/
theorem sound {name : String} {start : Nat} {tokens leftover : List Token} {decl : FunctionDecl}
    (scan : FunctionScan name start tokens decl leftover) :
    Decode.parseFunction name start tokens = .ok (decl, leftover) := by
  cases scan with
  | intro openParen parametersScan openBrace bodyScan =>
      unfold Decode.parseFunction
      rw [openParen]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseParameters _ _).bind _ = _
      rw [parametersScan.sound]
      change (Decode.expectKind .lbrace (Decode.dropNewlines _)).bind _ = _
      rw [openBrace]
      dsimp only [Decode.expectKind]
      rw [if_pos rfl]
      change (Decode.parseBlockFuel _ _).bind _ = _
      rw [bodyScan.sound]
      rfl

end FunctionScan

end Slang.Source.Parsing
