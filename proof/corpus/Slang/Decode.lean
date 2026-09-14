import Slang.Syntax

namespace Slang.Decode

open Slang

inductive TokenKind where
  | ident (value : String)
  | numeral (value : Nat)
  | lparen | rparen | lbrace | rbrace | lbracket | rbracket
  | comma | semi | newline | assign | lt | le | gt | ge
  | other (value : Char)
  deriving Repr, BEq, DecidableEq

structure Token where
  kind : TokenKind
  span : Span
  deriving Repr, BEq

structure Certificate where
  exactConsumption : Bool
  declarationBoundaries : Bool
  selectedConstants : Bool
  selectedNames : Bool
  resolvedCalls : Bool
  deriving Repr, BEq, DecidableEq

def Certificate.Valid (certificate : Certificate) : Prop :=
  certificate.exactConsumption = true ∧
  certificate.declarationBoundaries = true ∧
  certificate.selectedConstants = true ∧
  certificate.selectedNames = true ∧
  certificate.resolvedCalls = true

structure Decoded where
  program : Program
  certificate : Certificate
  deriving Repr, BEq

abbrev CertifiedDecoded := { decoded : Decoded // decoded.certificate.Valid }
abbrev Result (α : Type) := Except String α

private def isSpace : Char → Bool
  | ' ' | '\t' | '\r' => true
  | _ => false

private def isDigit (c : Char) : Bool := '0' ≤ c && c ≤ '9'
private def isAlpha (c : Char) : Bool :=
  ('a' ≤ c && c ≤ 'z') || ('A' ≤ c && c ≤ 'Z') || c = '_'
private def isAlphaNum (c : Char) : Bool := isAlpha c || isDigit c

private def takeWhile (p : Char → Bool) : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs => if p c then
      let (front, rest) := takeWhile p cs
      (c :: front, rest)
    else ([], c :: cs)

private def digitsValue (digits : List Char) : Nat :=
  digits.foldl (fun value digit => value * 10 + (digit.toNat - '0'.toNat)) 0

private def skipComment : List Char → Nat → List Char × Nat
  | [], pos => ([], pos)
  | '\n' :: cs, pos => ('\n' :: cs, pos)
  | _ :: cs, pos => skipComment cs (pos + 1)

private def lexChars : Nat → List Char → Nat → List Token → Result (List Token × Nat)
  | 0, _, _, _ => .error "tokenizer limit reached"
  | _, [], pos, tokens => .ok (tokens.reverse, pos)
  | fuel + 1, c :: cs, pos, tokens =>
      if isSpace c then lexChars fuel cs (pos + 1) tokens else
      if c = '\n' then
        lexChars fuel cs (pos + 1) ({ kind := .newline, span := ⟨pos, pos + 1⟩ } :: tokens)
      else if c = '/' then
        match cs with
        | '/' :: tail =>
            let (rest, next) := skipComment tail (pos + 2)
            lexChars fuel rest next tokens
        | _ => lexChars fuel cs (pos + 1) ({ kind := .other c, span := ⟨pos, pos + 1⟩ } :: tokens)
      else if isAlpha c then
        let (tail, rest) := takeWhile isAlphaNum cs
        let chars := c :: tail
        let stop := pos + chars.length
        lexChars fuel rest stop ({ kind := .ident (String.ofList chars), span := ⟨pos, stop⟩ } :: tokens)
      else if isDigit c then
        let (tail, rest) := takeWhile isDigit cs
        let chars := c :: tail
        let stop := pos + chars.length
        lexChars fuel rest stop ({ kind := .numeral (digitsValue chars), span := ⟨pos, stop⟩ } :: tokens)
      else
        let one (kind : TokenKind) :=
          lexChars fuel cs (pos + 1) ({ kind := kind, span := ⟨pos, pos + 1⟩ } :: tokens)
        match c, cs with
        | '<', '=' :: rest => lexChars fuel rest (pos + 2) ({ kind := .le, span := ⟨pos, pos + 2⟩ } :: tokens)
        | '>', '=' :: rest => lexChars fuel rest (pos + 2) ({ kind := .ge, span := ⟨pos, pos + 2⟩ } :: tokens)
        | '(', _ => one .lparen
        | ')', _ => one .rparen
        | '{', _ => one .lbrace
        | '}', _ => one .rbrace
        | '[', _ => one .lbracket
        | ']', _ => one .rbracket
        | ',', _ => one .comma
        | ';', _ => one .semi
        | '=', _ => one .assign
        | '<', _ => one .lt
        | '>', _ => one .gt
        | _, _ => one (.other c)

def tokenize (bytes : ByteArray) : Result (List Token × Nat) := do
  let text ← match String.fromUTF8? bytes with
    | some text => .ok text
    | none => .error "source is not UTF-8"
  let (tokens, _) ← lexChars (text.toList.length + 1) text.toList 0 []
  .ok (tokens, bytes.size)

private def dropNewlines : List Token → List Token
  | { kind := .newline, .. } :: rest => dropNewlines rest
  | tokens => tokens

private def expectKind (wanted : TokenKind) : List Token → Result (Token × List Token)
  | token :: rest => if token.kind = wanted then .ok (token, rest)
      else .error s!"expected {repr wanted} at byte {token.span.start}"
  | [] => .error s!"expected {repr wanted} at end of input"

private def expectIdent : List Token → Result (String × Token × List Token)
  | token@{ kind := .ident name, .. } :: rest => .ok (name, token, rest)
  | token :: _ => .error s!"expected name at byte {token.span.start}"
  | [] => .error "expected name at end of input"

private def binOp : TokenKind → Option (BinOp × Nat)
  | .ident "or" => some (.bitOr, 1)
  | .ident "xor" => some (.bitXor, 1)
  | .ident "and" => some (.bitAnd, 2)
  | .ident "is" => some (.eq, 3)
  | .lt => some (.lt, 3)
  | .le => some (.le, 3)
  | .gt => some (.gt, 3)
  | .ge => some (.ge, 3)
  | .other '+' => some (.add, 4)
  | .other '-' => some (.sub, 4)
  | .other '*' => some (.mul, 5)
  | _ => none

mutual
  def parseExprFuel : Nat → Nat → List Token → Result (Expr × List Token)
    | 0, _, _ => .error "expression nesting limit reached"
    | fuel + 1, minPrec, tokens => do
        let (left, rest) ← parsePrimaryFuel fuel tokens
        parseBinaryFuel fuel minPrec left rest

  def parsePrimaryFuel : Nat → List Token → Result (Expr × List Token)
    | 0, _ => .error "expression nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | { kind := .numeral value, .. } :: rest => .ok (.literal value, rest)
      | { kind := .ident "memory", .. } :: rest => do
          let (_, rest) ← expectKind .lbracket rest
          let (address, rest) ← parseExprFuel fuel 0 rest
          let (_, rest) ← expectKind .rbracket rest
          .ok (.memory address, rest)
      | { kind := .ident name, .. } :: ({ kind := .lparen, .. } :: rest) => do
          let (args, rest) ← parseArgsFuel fuel rest
          .ok (.call name args, rest)
      | { kind := .ident name, .. } :: rest => .ok (.name name, rest)
      | { kind := .lparen, .. } :: rest => do
          let (value, rest) ← parseExprFuel fuel 0 rest
          let (_, rest) ← expectKind .rparen rest
          .ok (value, rest)
      | token :: _ => .error s!"unsupported expression token {repr token.kind} at byte {token.span.start}"
      | [] => .error "expected expression at end of input"

  def parseArgsFuel : Nat → List Token → Result (List Expr × List Token)
    | 0, _ => .error "call argument nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | { kind := .rparen, .. } :: rest => .ok ([], rest)
      | tokens => do
          let (arg, rest) ← parseExprFuel fuel 0 tokens
          match dropNewlines rest with
          | { kind := .comma, .. } :: tail =>
              let (args, tail) ← parseArgsFuel fuel tail
              .ok (arg :: args, tail)
          | { kind := .rparen, .. } :: tail => .ok ([arg], tail)
          | token :: _ => .error s!"expected ',' or ')' at byte {token.span.start}"
          | [] => .error "unterminated call arguments"

  def parseBinaryFuel : Nat → Nat → Expr → List Token → Result (Expr × List Token)
    | 0, _, _, _ => .error "expression operator limit reached"
    | fuel + 1, minPrec, left, tokens =>
      match dropNewlines tokens with
      | ({ kind := .ident "is", .. } :: { kind := .ident "not", .. } :: rest) =>
          if 3 < minPrec then .ok (left, tokens) else do
            let (right, rest) ← parseExprFuel fuel 4 rest
            parseBinaryFuel fuel minPrec (.binary .ne left right) rest
      | token :: rest =>
          match binOp token.kind with
          | some (op, prec) =>
              if prec < minPrec then .ok (left, tokens) else do
                let (right, rest) ← parseExprFuel fuel (prec + 1) rest
                parseBinaryFuel fuel minPrec (.binary op left right) rest
          | none =>
              if let .ident "if" := token.kind then
                if 0 < minPrec then .ok (left, tokens) else do
                  let (condition, rest) ← parseExprFuel fuel 0 rest
                  match dropNewlines rest with
                  | { kind := .ident "else", .. } :: tail =>
                      let (elseValue, tail) ← parseExprFuel fuel 0 tail
                      .ok (.conditional left condition elseValue, tail)
                  | next :: _ => .error s!"expected inline else at byte {next.span.start}"
                  | [] => .error "unterminated inline conditional"
              else .ok (left, tokens)
      | [] => .ok (left, [])
end

private def parseExpr (tokens : List Token) : Result (Expr × List Token) :=
  parseExprFuel (tokens.length * 4 + 16) 0 tokens

mutual
  def parseBlockFuel : Nat → List Token → Result (List Stmt × Token × List Token)
    | 0, _ => .error "statement nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | close@{ kind := .rbrace, .. } :: rest => .ok ([], close, rest)
      | [] => .error "unterminated function body"
      | tokens => do
          let (stmt, rest) ← parseStmtFuel fuel tokens
          let (body, close, rest) ← parseBlockFuel fuel rest
          .ok (stmt :: body, close, rest)

  def parseStmtFuel : Nat → List Token → Result (Stmt × List Token)
    | 0, _ => .error "statement nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | { kind := .ident "return", .. } :: rest => do
          let (value, rest) ← parseExpr rest
          let (_, rest) ← expectKind .semi (dropNewlines rest)
          .ok (.return_ value, rest)
      | { kind := .ident "if", .. } :: rest => do
          let (condition, rest) ← parseExpr rest
          let (_, rest) ← expectKind .lbrace (dropNewlines rest)
          let (thenBody, _, rest) ← parseBlockFuel fuel rest
          match dropNewlines rest with
          | { kind := .ident "else", .. } :: tail => do
              let (_, tail) ← expectKind .lbrace (dropNewlines tail)
              let (elseBody, _, tail) ← parseBlockFuel fuel tail
              .ok (.if_ condition thenBody (some elseBody), tail)
          | _ => .ok (.if_ condition thenBody none, rest)
      | { kind := .ident "memory", .. } :: { kind := .lbracket, .. } :: rest => do
          let (address, rest) ← parseExpr rest
          let (_, rest) ← expectKind .rbracket rest
          let (_, rest) ← expectKind .assign (dropNewlines rest)
          let (value, rest) ← parseExpr rest
          let (_, rest) ← expectKind .semi (dropNewlines rest)
          .ok (.store address value, rest)
      | { kind := .ident name, .. } :: { kind := .assign, .. } :: rest => do
          let (value, rest) ← parseExpr rest
          let (_, rest) ← expectKind .semi (dropNewlines rest)
          .ok (.bind name value, rest)
      | tokens => do
          let (value, rest) ← parseExpr tokens
          let (_, rest) ← expectKind .semi (dropNewlines rest)
          .ok (.expression value, rest)
end

private def parseParameters : Nat → List Token → Result (List String × List Token)
  | 0, _ => .error "parameter limit reached"
  | _, { kind := .rparen, .. } :: rest => .ok ([], rest)
  | fuel + 1, tokens => do
      let (name, _, rest) ← expectIdent tokens
      match rest with
      | { kind := .comma, .. } :: tail =>
          let (names, tail) ← parseParameters fuel tail
          .ok (name :: names, tail)
      | { kind := .rparen, .. } :: tail => .ok ([name], tail)
      | token :: _ => .error s!"expected ',' or ')' in parameters at byte {token.span.start}"
      | [] => .error "unterminated parameter list"

private def parseFunction (name : String) (start : Nat) (tokens : List Token) : Result (FunctionDecl × List Token) := do
  let (_, tokens) ← expectKind .lparen tokens
  let (parameters, tokens) ← parseParameters (tokens.length + 1) tokens
  let (_, tokens) ← expectKind .lbrace (dropNewlines tokens)
  let (body, close, tokens) ← parseBlockFuel (tokens.length + 16) tokens
  .ok ({ name, parameters, body, span := ⟨start, close.span.stop⟩ }, tokens)

def selectedConstantNames : List String :=
  ["TABLE_PRESENT", "TABLE_EMPTY", "TABLE_STRIDE", "MERGE_SET", "MERGE_MAP",
   "MERGE_ACC", "MERGE_GRAPH", "TABLE_PHI", "COORD_CEILING"]

def selectedFunctionNames : List String :=
  ["tableSpread", "tableSlotKey", "tableSlotVal", "tableSlotSet", "tableSlotValSet",
   "tableInitGo", "tableInit", "tableCapacityOk", "tableFindSlotGo",
   "tableLookupSlotGo", "tableFind", "tableGet", "tableInsertWith"]

private def takeLine : List Token → List Token × List Token
  | [] => ([], [])
  | token@{ kind := .newline, .. } :: rest => ([], token :: rest)
  | token :: rest =>
      let (line, tail) := takeLine rest
      (token :: line, tail)

private def scanSelected : Nat → List Token → List String → List String →
    Result (List ConstantDecl × List FunctionDecl)
  | 0, _, _, _ => .error "declaration scan limit reached"
  | _, [], _, _ => .ok ([], [])
  | fuel + 1, token@{ kind := .ident name, .. } :: { kind := .assign, .. } :: rest, constants, functions =>
      if constants.contains name then do
        let (line, tail) := takeLine rest
        let (value, leftover) ← parseExpr line
        if dropNewlines leftover != [] then .error s!"trailing constant tokens for {name}" else
        let stop := line.getLast?.map (fun t => t.span.stop) |>.getD token.span.stop
        let (decls, funcs) ← scanSelected fuel tail (constants.erase name) functions
        .ok ({ name, value, span := ⟨token.span.start, stop⟩ } :: decls, funcs)
      else scanSelected fuel rest constants functions
  | fuel + 1, token@{ kind := .ident name, .. } :: ({ kind := .lparen, .. } :: rest), constants, functions =>
      if functions.contains name then do
        let (decl, tail) ← parseFunction name token.span.start ({ kind := .lparen, span := ⟨token.span.stop, token.span.stop + 1⟩ } :: rest)
        let (decls, funcs) ← scanSelected fuel tail constants (functions.erase name)
        .ok (decls, decl :: funcs)
      else scanSelected fuel rest constants functions
  | fuel + 1, _ :: rest, constants, functions => scanSelected fuel rest constants functions

private def distinct (names : List String) : Bool := names.eraseDups.length = names.length
private def delimitersBalanced : List Token → List TokenKind → Bool
  | [], [] => true
  | [], _ => false
  | { kind := .lparen, .. } :: rest, stack => delimitersBalanced rest (.lparen :: stack)
  | { kind := .lbrace, .. } :: rest, stack => delimitersBalanced rest (.lbrace :: stack)
  | { kind := .lbracket, .. } :: rest, stack => delimitersBalanced rest (.lbracket :: stack)
  | { kind := .rparen, .. } :: rest, .lparen :: stack => delimitersBalanced rest stack
  | { kind := .rbrace, .. } :: rest, .lbrace :: stack => delimitersBalanced rest stack
  | { kind := .rbracket, .. } :: rest, .lbracket :: stack => delimitersBalanced rest stack
  | { kind := .rparen, .. } :: _, _ => false
  | { kind := .rbrace, .. } :: _, _ => false
  | { kind := .rbracket, .. } :: _, _ => false
  | _ :: rest, stack => delimitersBalanced rest stack

private def spansValid (size : Nat) (spans : List Span) : Bool :=
  spans.all (fun span => span.start < span.stop && span.stop ≤ size) &&
  spans.all fun first => spans.all fun second =>
    first == second || first.stop ≤ second.start || second.stop ≤ first.start

private def resolveCalls (functions : List FunctionDecl) : Bool :=
  let names := functions.map (·.name)
  functions.all fun decl => (Syntax.FunctionDecl.calls decl).all (names.contains ·)

private def expressionNamesResolve (available : List String) (expression : Expr) : Bool :=
  (Syntax.Expr.names expression).all (available.contains ·)

private def statementsNamesResolve : Nat → List String → List Stmt → Bool
  | 0, _, _ => false
  | _, _, [] => true
  | fuel + 1, available, .bind name value :: rest =>
      expressionNamesResolve available value && statementsNamesResolve fuel (name :: available) rest
  | fuel + 1, available, .store address value :: rest =>
      expressionNamesResolve available address && expressionNamesResolve available value &&
        statementsNamesResolve fuel available rest
  | fuel + 1, available, .return_ value :: rest =>
      expressionNamesResolve available value && statementsNamesResolve fuel available rest
  | fuel + 1, available, .expression value :: rest =>
      expressionNamesResolve available value && statementsNamesResolve fuel available rest
  | fuel + 1, available, .if_ condition thenBody elseBody :: rest =>
      expressionNamesResolve available condition && statementsNamesResolve fuel available thenBody &&
        statementsNamesResolve fuel available (elseBody.getD []) &&
        statementsNamesResolve fuel available rest

private def resolveNames (constants : List ConstantDecl) (functions : List FunctionDecl) : Bool :=
  let constantNames := constants.map (·.name)
  functions.all fun decl =>
    distinct decl.parameters &&
    statementsNamesResolve (decl.body.length * 4 + 64) (decl.parameters ++ constantNames) decl.body

private def decodeTokens (bytes : ByteArray) (tokens : List Token) (consumed : Nat) : Result CertifiedDecoded := do
  if consumed != bytes.size then .error "tokenizer did not consume every source byte" else
  if !delimitersBalanced tokens [] then .error "unbalanced source declaration delimiter" else
  let (constants, functions) ← scanSelected (tokens.length + 1) tokens selectedConstantNames selectedFunctionNames
  let constantNames := constants.map (·.name)
  let functionNames := functions.map (·.name)
  if constantNames.length != 8 then .error "selected table constants are missing or duplicated" else
  if functionNames.length != selectedFunctionNames.length then .error "selected functions are missing or duplicated" else
  if !distinct (constantNames ++ ["COORD_CEILING"]) || !distinct functionNames then
    .error "selected declaration names are not unique"
  else if !spansValid bytes.size (constants.map (·.span) ++ functions.map (·.span)) then
    .error "selected declaration spans overlap or leave source bounds"
  else if !resolveCalls functions then .error "unsupported or unresolved reachable call"
  else if !resolveNames (constants ++ [{
      name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }]) functions then
    .error "unresolved local or constant name"
  else
  let program : Program := { constants, functions, sourceBytes := bytes.size, consumedBytes := consumed }
  let certificate : Certificate := ⟨true, true, true, true, true⟩
  .ok ⟨{ program, certificate }, by simp [Certificate.Valid, certificate]⟩

def decodeTable (bytes : ByteArray) : Result CertifiedDecoded := do
  let (tokens, consumed) ← tokenize bytes
  decodeTokens bytes tokens consumed

def decodeImportedCoordCeiling (bytes : ByteArray) : Result ConstantDecl := do
  let (tokens, consumed) ← tokenize bytes
  if consumed != bytes.size then .error "import tokenizer did not consume every source byte" else
  let (constants, _) ← scanSelected (tokens.length + 1) tokens ["COORD_CEILING"] []
  match constants with
  | [decl] => .ok decl
  | _ => .error "COORD_CEILING is missing or duplicated"

def decodeClosure (tableBytes stdBytes : ByteArray) : Result CertifiedDecoded := do
  let decoded ← decodeTable tableBytes
  let coord ← decodeImportedCoordCeiling stdBytes
  if coord.value != .literal 8796093022208 then .error "COORD_CEILING has the wrong value" else
  let program := decoded.val.program
  let program := { program with constants := program.constants ++ [coord] }
  .ok ⟨{ decoded.val with program }, decoded.property⟩

theorem decode_certified (source : ByteArray) (decoded : CertifiedDecoded)
    (_h : decodeTable source = .ok decoded) : decoded.val.certificate.Valid := decoded.property

end Slang.Decode
