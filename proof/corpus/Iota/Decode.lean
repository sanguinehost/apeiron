import Iota.Syntax

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Decode

open Iota

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

def isSpace : Char → Bool
  | ' ' | '\t' | '\r' => true
  | _ => false

def isDigit (c : Char) : Bool :=
  '0'.toNat ≤ c.toNat && c.toNat ≤ '9'.toNat
def isAlpha (c : Char) : Bool :=
  ('a'.toNat ≤ c.toNat && c.toNat ≤ 'z'.toNat) ||
    ('A'.toNat ≤ c.toNat && c.toNat ≤ 'Z'.toNat) || c.toNat == '_'.toNat
def isAlphaNum (c : Char) : Bool := isAlpha c || isDigit c

-- Temporal O(n) for n input characters: at most one `p` application per character. Spatial O(n)
-- stack: `c :: front` binds the recursive result, so one frame per character visited.
def takeWhile (p : Char → Bool) : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs => if p c then
      let (front, rest) := takeWhile p cs
      (c :: front, rest)
    else ([], c :: cs)

-- Temporal O(d) for d digits: one fold step (multiply, add, subtract) per digit. Spatial O(1): a
-- single Nat accumulator, `List.foldl` is tail-recursive.
def digitsValue (digits : List Char) : Nat :=
  digits.foldl (fun value digit => value * 10 + (digit.toNat - '0'.toNat)) 0

/-- Number of UTF-8 bytes occupied by a character. -/
def utf8ByteLength (c : Char) : Nat :=
  if c.toNat < 0x80 then 1
  else if c.toNat < 0x800 then 2
  else if c.toNat < 0x10000 then 3
  else 4

/-- Number of UTF-8 bytes occupied by a character list.

Temporal O(c) for c characters: one `utf8ByteLength` call per character. Spatial O(1): a single Nat
accumulator. -/
def utf8ByteLengthList (chars : List Char) : Nat :=
  chars.foldl (fun size c => size + utf8ByteLength c) 0

def isContinuationByte (value : Nat) : Bool :=
  0x80 ≤ value && value < 0xc0

-- Temporal O(fuel): each step decrements fuel by one and consumes 1–4 bytes, so fuel bounds the
-- steps. Spatial O(fuel) stack: the recursive tail is bound and consed, so one frame per step.
-- Operations counted: bytes read, continuation-byte tests.
def decodeUtf8CharsFuel : Nat → List UInt8 → Result (List Char)
  | 0, _ => .error "source is not UTF-8"
  | _ + 1, [] => .ok []
  | fuel + 1, first :: rest =>
      let b0 := first.toNat
      if b0 < 0x80 then do
        let tail ← decodeUtf8CharsFuel fuel rest
        .ok (Char.ofNat b0 :: tail)
      else if b0 < 0xe0 then
        match rest with
        | [] => .error "source is not UTF-8"
        | second :: rest =>
            let b1 := second.toNat
            let code := (b0 - 0xc0) * 0x40 + (b1 - 0x80)
            if b0 < 0xc2 || !isContinuationByte b1 then .error "source is not UTF-8"
            else do
              let tail ← decodeUtf8CharsFuel fuel rest
              .ok (Char.ofNat code :: tail)
      else if b0 < 0xf0 then
        match rest with
        | [] => .error "source is not UTF-8"
        | second :: afterSecond =>
            match afterSecond with
            | [] => .error "source is not UTF-8"
            | third :: rest =>
                let b1 := second.toNat
                let b2 := third.toNat
                let code := (b0 - 0xe0) * 0x1000 + (b1 - 0x80) * 0x40 + (b2 - 0x80)
                if !isContinuationByte b1 || !isContinuationByte b2 || code < 0x800 ||
                    (0xd800 ≤ code && code < 0xe000) then .error "source is not UTF-8"
                else do
                  let tail ← decodeUtf8CharsFuel fuel rest
                  .ok (Char.ofNat code :: tail)
      else if b0 < 0xf5 then
        match rest with
        | [] => .error "source is not UTF-8"
        | second :: afterSecond =>
            match afterSecond with
            | [] => .error "source is not UTF-8"
            | third :: afterThird =>
                match afterThird with
                | [] => .error "source is not UTF-8"
                | fourth :: rest =>
                    let b1 := second.toNat
                    let b2 := third.toNat
                    let b3 := fourth.toNat
                    let code := (b0 - 0xf0) * 0x40000 + (b1 - 0x80) * 0x1000 +
                      (b2 - 0x80) * 0x40 + (b3 - 0x80)
                    if !isContinuationByte b1 || !isContinuationByte b2 ||
                        !isContinuationByte b3 || code < 0x10000 || 0x110000 ≤ code then
                      .error "source is not UTF-8"
                    else do
                      let tail ← decodeUtf8CharsFuel fuel rest
                      .ok (Char.ofNat code :: tail)
      else .error "source is not UTF-8"

-- Temporal O(|bytes|): the fuel is |bytes| + 1 and each step consumes at least one byte. Spatial
-- O(|bytes|): the `decodeUtf8CharsFuel` stack plus the decoded character list.
def decodeUtf8Chars (bytes : List UInt8) : Result (List Char) :=
  decodeUtf8CharsFuel (bytes.length + 1) bytes

-- Temporal O(k) for the k characters before the next newline or end: one `utf8ByteLength` call per
-- character. Spatial O(1) call stack: the recursion is tail-recursive and shares the input tail.
def skipComment : List Char → Nat → List Char × Nat
  | [], pos => ([], pos)
  | '\n' :: cs, pos => ('\n' :: cs, pos)
  | c :: cs, pos => skipComment cs (pos + utf8ByteLength c)

-- Temporal O(n) for n input characters: each character is visited once by `lexChars`, `takeWhile` or
-- `skipComment`; one `utf8ByteLength` and one kind test per character. Spatial O(n): the token
-- accumulator and final `.reverse` are O(#tokens) and `takeWhile` adds O(longest run) transient
-- stack; the `lexChars` recursion itself is tail-recursive.
def lexChars : Nat → List Char → Nat → List Token → Result (List Token × Nat)
  | 0, _, _, _ => .error "tokenizer limit reached"
  | _ + 1, [], pos, tokens => .ok (tokens.reverse, pos)
  | fuel + 1, c :: cs, pos, tokens =>
      let width := utf8ByteLength c
      if isSpace c then lexChars fuel cs (pos + width) tokens else
      if c = '\n' then
        lexChars fuel cs (pos + width) ({ kind := .newline, span := ⟨pos, pos + width⟩ } :: tokens)
      else if c = '/' then
        match cs with
        | [] => lexChars fuel cs (pos + width) ({ kind := .other c, span := ⟨pos, pos + width⟩ } :: tokens)
        | second :: tail =>
            if second.toNat == '/'.toNat then
              let (rest, next) := skipComment tail (pos + 2)
              lexChars fuel rest next tokens
            else lexChars fuel cs (pos + width) ({ kind := .other c, span := ⟨pos, pos + width⟩ } :: tokens)
      else if isAlpha c then
        let (tail, rest) := takeWhile isAlphaNum cs
        let chars := c :: tail
        let stop := pos + utf8ByteLengthList chars
        lexChars fuel rest stop ({ kind := .ident (String.ofList chars), span := ⟨pos, stop⟩ } :: tokens)
      else if isDigit c then
        let (tail, rest) := takeWhile isDigit cs
        let chars := c :: tail
        let stop := pos + utf8ByteLengthList chars
        lexChars fuel rest stop ({ kind := .numeral (digitsValue chars), span := ⟨pos, stop⟩ } :: tokens)
      else
        let one (kind : TokenKind) :=
          lexChars fuel cs (pos + width) ({ kind := kind, span := ⟨pos, pos + width⟩ } :: tokens)
        if c.toNat == '<'.toNat then
          match cs with
          | [] => one .lt
          | second :: rest =>
              if second.toNat == '='.toNat then
                lexChars fuel rest (pos + 2) ({ kind := .le, span := ⟨pos, pos + 2⟩ } :: tokens)
              else one .lt
        else if c.toNat == '>'.toNat then
          match cs with
          | [] => one .gt
          | second :: rest =>
              if second.toNat == '='.toNat then
                lexChars fuel rest (pos + 2) ({ kind := .ge, span := ⟨pos, pos + 2⟩ } :: tokens)
              else one .gt
        else if c.toNat == '('.toNat then one .lparen
        else if c.toNat == ')'.toNat then one .rparen
        else if c.toNat == '{'.toNat then one .lbrace
        else if c.toNat == '}'.toNat then one .rbrace
        else if c.toNat == '['.toNat then one .lbracket
        else if c.toNat == ']'.toNat then one .rbracket
        else if c.toNat == ','.toNat then one .comma
        else if c.toNat == ';'.toNat then one .semi
        else if c.toNat == '='.toNat then one .assign
        else one (.other c)

-- Temporal O(n) for n source bytes: `decodeUtf8Chars` and `lexChars` (fuel n + 1) are each linear.
-- Spatial O(n): the decoded character list and the token list.
def tokenize (bytes : ByteArray) : Result (List Token × Nat) := do
  let chars ← decodeUtf8Chars bytes.data.toList
  lexChars (chars.length + 1) chars 0 []

-- Temporal O(k) for k leading newline tokens: one kind comparison per dropped token. Spatial O(1)
-- call stack: the recursion is tail-recursive and returns the original tail.
def dropNewlines : List Token → List Token
  | [] => []
  | token :: rest => if token.kind = .newline then dropNewlines rest else token :: rest

def expectKind (wanted : TokenKind) : List Token → Result (Token × List Token)
  | token :: rest => if token.kind = wanted then .ok (token, rest)
      else .error "unexpected token kind"
  | [] => .error "expected token at end of input"

private def expectIdent : List Token → Result (String × Token × List Token)
  | token@{ kind := .ident name, .. } :: rest => .ok (name, token, rest)
  | _ :: _ => .error "expected name"
  | [] => .error "expected name at end of input"

def binOp (kind : TokenKind) : Option (BinOp × Nat) :=
  if kind = .ident "or" then some (.bitOr, 1)
  else if kind = .ident "xor" then some (.bitXor, 1)
  else if kind = .ident "and" then some (.bitAnd, 2)
  else if kind = .ident "is" then some (.eq, 3)
  else if kind = .lt then some (.lt, 3)
  else if kind = .le then some (.le, 3)
  else if kind = .gt then some (.gt, 3)
  else if kind = .ge then some (.ge, 3)
  else if kind = .other '+' then some (.add, 4)
  else if kind = .other '-' then some (.sub, 4)
  else if kind = .other '*' then some (.mul, 5)
  else none

mutual
  -- Temporal O(fuel): every recursive call passes a strictly smaller fuel and each descent consumes
  -- tokens, so fuel bounds the number of steps; entry points set fuel linear in the token count.
  -- Spatial O(fuel) stack: recursive results are bound (`let (left, rest) ← ...`), so one frame per
  -- fuel level. Operations counted: tokens consumed, `dropNewlines`/`binOp` comparisons.
  def parseExprFuel : Nat → Nat → List Token → Result (Expr × List Token)
    | 0, _, _ => .error "expression nesting limit reached"
    | fuel + 1, minPrec, tokens => do
        let (left, rest) ← parsePrimaryFuel fuel tokens
        parseBinaryFuel fuel minPrec left rest

  -- Temporal O(fuel) and spatial O(fuel) stack, as `parseExprFuel`: each branch consumes tokens and
  -- descends with a smaller fuel; the `dropNewlines` scan is charged to the tokens it skips.
  def parsePrimaryFuel : Nat → List Token → Result (Expr × List Token)
    | 0, _ => .error "expression nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | [] => .error "expected expression at end of input"
      | token :: rest =>
          match token.kind with
          | .numeral value => .ok (.literal value, rest)
          | .ident name =>
              if name = "memory" then do
                let (_, rest) ← expectKind .lbracket rest
                let (address, rest) ← parseExprFuel fuel 0 rest
                let (_, rest) ← expectKind .rbracket rest
                .ok (.memory address, rest)
              else
                match rest with
                | next :: tail =>
                    if next.kind = .lparen then do
                      let (args, rest) ← parseArgsFuel fuel tail
                      .ok (.call name args, rest)
                    else .ok (.name name, rest)
                | [] => .ok (.name name, [])
          | .lparen => do
              let (value, rest) ← parseExprFuel fuel 0 rest
              let (_, rest) ← expectKind .rparen rest
              .ok (value, rest)
          | .rparen | .lbrace | .rbrace | .lbracket | .rbracket | .comma | .semi |
              .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
              .error "unsupported expression token"

  -- Temporal O(fuel): each argument consumes at least one token and recurses with a smaller fuel.
  -- Spatial O(fuel) stack: `arg :: args` binds the recursive result, so one frame per argument.
  def parseArgsFuel : Nat → List Token → Result (List Expr × List Token)
    | 0, _ => .error "call argument nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | [] => .error "unterminated call arguments"
      | token :: rest =>
          if token.kind = .rparen then .ok ([], rest) else do
            let (arg, rest) ← parseExprFuel fuel 0 (token :: rest)
            match dropNewlines rest with
            | next :: tail =>
                if next.kind = .comma then
                  let (args, tail) ← parseArgsFuel fuel tail
                  .ok (arg :: args, tail)
                else if next.kind = .rparen then .ok ([arg], tail)
                else .error "expected comma or closing parenthesis"
            | [] => .error "unterminated call arguments"

  -- Temporal O(fuel): each operator consumes at least one token and recurses with a smaller fuel.
  -- Spatial O(fuel) stack: the recursive result is bound before the next step, so one frame per
  -- operator. Operations counted: tokens consumed, `binOp` comparisons.
  def parseBinaryFuel : Nat → Nat → Expr → List Token → Result (Expr × List Token)
    | 0, _, _, _ => .error "expression operator limit reached"
    | fuel + 1, minPrec, left, tokens =>
      match dropNewlines tokens with
      | [] => .ok (left, [])
      | token :: rest =>
          let ordinary :=
            match binOp token.kind with
            | some (op, prec) =>
                if prec < minPrec then .ok (left, tokens) else do
                  let (right, rest) ← parseExprFuel fuel (prec + 1) rest
                  parseBinaryFuel fuel minPrec (.binary op left right) rest
            | none =>
                if token.kind = .ident "if" then
                  if 0 < minPrec then .ok (left, tokens) else do
                    let (condition, rest) ← parseExprFuel fuel 0 rest
                    match dropNewlines rest with
                    | next :: tail =>
                        if next.kind = .ident "else" then
                          let (elseValue, tail) ← parseExprFuel fuel 0 tail
                          .ok (.conditional left condition elseValue, tail)
                        else .error "expected inline else"
                    | [] => .error "unterminated inline conditional"
                else .ok (left, tokens)
          if token.kind = .ident "is" then
            match rest with
            | next :: tail =>
                if next.kind = .ident "not" then
                  if 3 < minPrec then .ok (left, tokens) else do
                    let (right, rest) ← parseExprFuel fuel 4 tail
                    parseBinaryFuel fuel minPrec (.binary .ne left right) rest
                else ordinary
            | [] => ordinary
          else ordinary
end

-- Temporal O(t) for t = |tokens|: fuel is `t * 4 + 16` and bounds the parser steps. Spatial O(t)
-- stack: one frame per fuel level in `parseExprFuel`.
private def parseExpr (tokens : List Token) : Result (Expr × List Token) :=
  parseExprFuel (tokens.length * 4 + 16) 0 tokens

mutual
  -- Temporal O(fuel): each statement consumes tokens and recurses with a smaller fuel; the
  -- `parseExpr` calls are charged to the tokens they consume. Spatial O(fuel) stack: the body list
  -- is built from the bound recursive result, so one frame per statement.
  def parseBlockFuel : Nat → List Token → Result (List Stmt × Token × List Token)
    | 0, _ => .error "statement nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | [] => .error "unterminated function body"
      | token :: rest =>
          if token.kind = .rbrace then .ok ([], token, rest) else do
            let (stmt, rest) ← parseStmtFuel fuel (token :: rest)
            let (body, close, rest) ← parseBlockFuel fuel rest
            .ok (stmt :: body, close, rest)

  -- Temporal O(fuel): fuel bounds the statement count and each `parseExpr` consumes its own tokens.
  -- Spatial O(fuel) stack: recursive results are bound, so one frame per nested block/statement.
  def parseStmtFuel : Nat → List Token → Result (Stmt × List Token)
    | 0, _ => .error "statement nesting limit reached"
    | fuel + 1, tokens =>
      match dropNewlines tokens with
      | token :: rest =>
          match token.kind with
          | .ident name =>
              if name = "return" then do
                let (value, rest) ← parseExpr rest
                let (_, rest) ← expectKind .semi (dropNewlines rest)
                .ok (.return_ value, rest)
              else if name = "if" then do
                let (condition, rest) ← parseExpr rest
                let (_, rest) ← expectKind .lbrace (dropNewlines rest)
                let (thenBody, _, rest) ← parseBlockFuel fuel rest
                match dropNewlines rest with
                | next :: tail =>
                    if next.kind = .ident "else" then do
                      let (_, tail) ← expectKind .lbrace (dropNewlines tail)
                      let (elseBody, _, tail) ← parseBlockFuel fuel tail
                      .ok (.if_ condition thenBody (some elseBody), tail)
                    else .ok (.if_ condition thenBody none, rest)
                | [] => .ok (.if_ condition thenBody none, rest)
              else if name = "memory" then
                match rest with
                | next :: tail =>
                    if next.kind = .lbracket then do
                      let (address, rest) ← parseExpr tail
                      let (_, rest) ← expectKind .rbracket rest
                      let (_, rest) ← expectKind .assign (dropNewlines rest)
                      let (value, rest) ← parseExpr rest
                      let (_, rest) ← expectKind .semi (dropNewlines rest)
                      .ok (.store address value, rest)
                    else do
                      let (value, rest) ← parseExpr (token :: next :: tail)
                      let (_, rest) ← expectKind .semi (dropNewlines rest)
                      .ok (.expression value, rest)
                | [] => do
                    let (value, rest) ← parseExpr [token]
                    let (_, rest) ← expectKind .semi (dropNewlines rest)
                    .ok (.expression value, rest)
              else
                match rest with
                | next :: tail =>
                    if next.kind = .assign then do
                      let (value, rest) ← parseExpr tail
                      let (_, rest) ← expectKind .semi (dropNewlines rest)
                      .ok (.bind name value, rest)
                    else do
                      let (value, rest) ← parseExpr (token :: next :: tail)
                      let (_, rest) ← expectKind .semi (dropNewlines rest)
                      .ok (.expression value, rest)
                | [] => do
                    let (value, rest) ← parseExpr [token]
                    let (_, rest) ← expectKind .semi (dropNewlines rest)
                    .ok (.expression value, rest)
          | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
              .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ => do
              let (value, rest) ← parseExpr (token :: rest)
              let (_, rest) ← expectKind .semi (dropNewlines rest)
              .ok (.expression value, rest)
      | [] => do
          let (value, rest) ← parseExpr []
          let (_, rest) ← expectKind .semi (dropNewlines rest)
          .ok (.expression value, rest)
end

-- Temporal O(fuel): one token (and one fuel decrement) per parameter. Spatial O(fuel) stack:
-- `name :: names` binds the recursive result, so one frame per parameter.
def parseParameters : Nat → List Token → Result (List String × List Token)
  | 0, _ => .error "parameter limit reached"
  | _ + 1, [] => .error "expected name at end of input"
  | fuel + 1, token :: rest =>
      if token.kind = .rparen then .ok ([], rest) else
      match token.kind with
      | .ident name =>
          match rest with
          | [] => .error "unterminated parameter list"
          | next :: tail =>
              if next.kind = .comma then do
                let (names, tail) ← parseParameters fuel tail
                .ok (name :: names, tail)
              else if next.kind = .rparen then .ok ([name], tail)
              else .error "expected ',' or ')' in parameters"
      | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
          .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
          .error "expected name"

-- Temporal O(t) for t = |tokens|: `parseParameters` and `parseBlockFuel` are each run with fuel
-- linear in the remaining token count. Spatial O(t) stack from those recursions.
def parseFunction (name : String) (start : Nat) (tokens : List Token) : Result (FunctionDecl × List Token) := do
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

-- Temporal O(k) for the k tokens before the next newline: one kind comparison per token. Spatial
-- O(k) stack: `token :: line` binds the recursive result, so one frame per token.
def takeLine : List Token → List Token × List Token
  | [] => ([], [])
  | token :: rest =>
      if token.kind = .newline then ([], token :: rest) else
        let (line, tail) := takeLine rest
        (token :: line, tail)

-- Temporal O(fuel) for fuel = |tokens| + 1: each step consumes one token and recurses with a smaller
-- fuel; `takeLine`/`parseExpr`/`parseFunction` work is charged to the tokens they consume, and
-- `contains`/`erase` scan the bounded (≤13) selected-name lists, so O(1) each. Spatial O(fuel)
-- stack: recursive results are bound, plus the `takeLine`/`parseFunction` stacks.
def scanSelected : Nat → List Token → List String → List String →
    Result (List ConstantDecl × List FunctionDecl)
  | 0, _, _, _ => .error "declaration scan limit reached"
  | _, [], _, _ => .ok ([], [])
  | fuel + 1, token :: rest, constants, functions =>
      match token.kind with
      | .ident name =>
          match rest with
          | [] => scanSelected fuel [] constants functions
          | next :: tail =>
              if next.kind = .assign then
                if constants.contains name then do
                  let (line, afterLine) := takeLine tail
                  let (value, leftover) ← parseExpr line
                  if dropNewlines leftover != [] then
                    .error "trailing constant tokens"
                  else
                    let stop := line.getLast?.map (fun t => t.span.stop) |>.getD token.span.stop
                    let (decls, funcs) ←
                      scanSelected fuel afterLine (constants.erase name) functions
                    .ok ({ name, value, span := ⟨token.span.start, stop⟩ } :: decls, funcs)
                else if selectedConstantNames.contains name then
                  .error "duplicate selected constant"
                else scanSelected fuel rest constants functions
              else if next.kind = .lparen then
                if functions.contains name then do
                  let openParen : Token :=
                    { kind := .lparen, span := ⟨token.span.stop, token.span.stop + 1⟩ }
                  let (decl, afterDecl) ← parseFunction name token.span.start (openParen :: tail)
                  let (decls, funcs) ←
                    scanSelected fuel afterDecl constants (functions.erase name)
                  .ok (decls, decl :: funcs)
                else scanSelected fuel rest constants functions
              else scanSelected fuel rest constants functions
      | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
          .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
          scanSelected fuel rest constants functions

-- Temporal O(s²) for s = |names|: `List.eraseDups` is O(s²) BEq comparisons (its own doc) and both
-- `.length` are O(s). Spatial O(s) for the erased list (the erase loop is tail-recursive).
def distinct (names : List String) : Bool := names.eraseDups.length = names.length
-- Temporal O(t) for t = |tokens|: one token visited per step with one TokenKind comparison on a
-- closing delimiter. Spatial O(d) for the `stack` accumulator at nesting depth d (the recursion is
-- tail-recursive, so O(1) call stack).
def delimitersBalanced : List Token → List TokenKind → Bool
  | tokens, stack =>
      match tokens with
      | [] =>
          match stack with
          | [] => true
          | _ :: _ => false
      | token :: rest =>
          match token.kind with
          | .lparen => delimitersBalanced rest (.lparen :: stack)
          | .lbrace => delimitersBalanced rest (.lbrace :: stack)
          | .lbracket => delimitersBalanced rest (.lbracket :: stack)
          | .rparen =>
              match stack with
              | [] => false
              | top :: tail =>
                  if top = .lparen then delimitersBalanced rest tail else false
          | .rbrace =>
              match stack with
              | [] => false
              | top :: tail =>
                  if top = .lbrace then delimitersBalanced rest tail else false
          | .rbracket =>
              match stack with
              | [] => false
              | top :: tail =>
                  if top = .lbracket then delimitersBalanced rest tail else false
          | .ident _ | .numeral _ | .comma | .semi | .newline | .assign | .lt | .le |
              .gt | .ge | .other _ => delimitersBalanced rest stack

-- Temporal O(s²) for s = |spans|: the first `all` is O(s) endpoint checks and the nested `all` is
-- O(s²) pairwise overlap comparisons. Spatial O(1) beyond the input: both `all`s are folds.
def spansValid (size : Nat) (spans : List Span) : Bool :=
  spans.all (fun span => span.start < span.stop && span.stop ≤ size) &&
  spans.all fun first => spans.all fun second =>
    first == second || first.stop ≤ second.start || second.stop ≤ first.start

mutual
  -- Temporal O(e · |available|) for e expression nodes: one visit per node and one `List.contains`
  -- scan of `available` at each `.call`. Spatial O(e) stack: each sub-result is bound before the
  -- parent combines it.
  def expressionCallsResolve (available : List String) : Expr → Bool
    | .literal _ | .name _ => true
    | .call callee args =>
        available.contains callee && expressionsCallsResolve available args
    | .memory address => expressionCallsResolve available address
    | .binary _ left right =>
        expressionCallsResolve available left && expressionCallsResolve available right
    | .conditional thenValue condition elseValue =>
        expressionCallsResolve available thenValue &&
          expressionCallsResolve available condition &&
          expressionCallsResolve available elseValue

  -- Temporal O(e · |available|) for e list elements, as `expressionCallsResolve`. Spatial O(e)
  -- stack: `&&` binds the head result before the tail recursion.
  def expressionsCallsResolve (available : List String) : List Expr → Bool
    | [] => true
    | expression :: rest =>
        expressionCallsResolve available expression && expressionsCallsResolve available rest
end

mutual
  -- Temporal O(s · |available|) for s statement nodes, as `expressionCallsResolve`. Spatial O(s)
  -- stack: each sub-result is bound before the parent combines it.
  def statementCallsResolve (available : List String) : Stmt → Bool
    | .bind _ value => expressionCallsResolve available value
    | .store address value =>
        expressionCallsResolve available address && expressionCallsResolve available value
    | .return_ value => expressionCallsResolve available value
    | .expression value => expressionCallsResolve available value
    | .if_ condition thenBody elseBody =>
        expressionCallsResolve available condition &&
          statementsCallsResolve available thenBody &&
          match elseBody with
          | none => true
          | some body => statementsCallsResolve available body

  -- Temporal O(s · |available|) for s list elements, as `statementCallsResolve`. Spatial O(s)
  -- stack: `&&` binds the head result before the tail recursion.
  def statementsCallsResolve (available : List String) : List Stmt → Bool
    | [] => true
    | stmt :: rest =>
        statementCallsResolve available stmt && statementsCallsResolve available rest
end

-- Temporal O(Σ |body| + |functions|): one `functions.map` over the names, then one
-- `statementsCallsResolve` per body charged to its statement/expression nodes. Spatial O(|functions|
-- + maximum body depth) for the name list and the recursion.
def resolveCalls (functions : List FunctionDecl) : Bool :=
  let names := functions.map (·.name)
  functions.all fun decl => statementsCallsResolve names decl.body

mutual
  -- Temporal O(e · |available|) for e expression nodes: one visit per node and one `available.contains`
  -- scan at each `.name`. Spatial O(e) stack: each sub-result is bound before the parent combines it.
  def expressionNamesResolve (available : List String) : Expr → Bool
    | .literal _ => true
    | .name name => available.contains name
    | .call _ args => expressionsNamesResolve available args
    | .memory address => expressionNamesResolve available address
    | .binary _ left right =>
        expressionNamesResolve available left && expressionNamesResolve available right
    | .conditional thenValue condition elseValue =>
        expressionNamesResolve available thenValue &&
          expressionNamesResolve available condition &&
          expressionNamesResolve available elseValue

  -- Temporal O(e · |available|) for e list elements, as `expressionNamesResolve`. Spatial O(e)
  -- stack: `&&` binds the head result before the tail recursion.
  def expressionsNamesResolve (available : List String) : List Expr → Bool
    | [] => true
    | expression :: rest =>
        expressionNamesResolve available expression && expressionsNamesResolve available rest
end

-- Temporal O(fuel · (|available| + fuel)): fuel bounds the statements visited, each `.name`/`.call`
-- lookup scans `available`, and `.bind` grows `available` by at most fuel. Spatial O(fuel) stack:
-- recursive results are bound and each `.bind` conses one name.
def statementsNamesResolve : Nat → List String → List Stmt → Bool
  | 0, _, _ => false
  | _, _, [] => true
  | fuel + 1, available, stmt :: rest =>
      match stmt with
      | .bind name value =>
          expressionNamesResolve available value &&
            statementsNamesResolve fuel (name :: available) rest
      | .store address value =>
          expressionNamesResolve available address && expressionNamesResolve available value &&
            statementsNamesResolve fuel available rest
      | .return_ value =>
          expressionNamesResolve available value && statementsNamesResolve fuel available rest
      | .expression value =>
          expressionNamesResolve available value && statementsNamesResolve fuel available rest
      | .if_ condition thenBody elseBody =>
          expressionNamesResolve available condition &&
            statementsNamesResolve fuel available thenBody &&
            (match elseBody with
              | none => true
              | some body => statementsNamesResolve fuel available body) &&
            statementsNamesResolve fuel available rest

-- Temporal O(Σ_functions (|params|² + |body|² + |body| · |available|)): `distinct` is quadratic in
-- the parameter list and `statementsNamesResolve` is as above per body, with `available` the
-- parameters plus all constants. Spatial O(maximum body depth + Σ |params|).
def resolveNames (constants : List ConstantDecl) (functions : List FunctionDecl) : Bool :=
  let constantNames := constants.map (·.name)
  functions.all fun decl =>
    distinct decl.parameters &&
    statementsNamesResolve (decl.body.length * 4 + 64) (decl.parameters ++ constantNames) decl.body

-- Temporal O(t + program size): linear scans over the t tokens (`delimitersBalanced`, `scanSelected`),
-- then the call/name resolution over the selected declarations; `distinct`/`spansValid` run over the
-- fixed ≤22 selected declarations, so their quadratic terms are a constant. Spatial O(t) for the
-- token and declaration lists plus the resolution stacks.
def decodeTokens (bytes : ByteArray) (tokens : List Token) (consumed : Nat) : Result CertifiedDecoded := do
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
  .ok ⟨{ program, certificate }, ⟨rfl, rfl, rfl, rfl, rfl⟩⟩

-- Temporal O(n) for n source bytes: `tokenize` is linear and `decodeTokens` runs over the O(n)
-- tokens. Spatial O(n): the decoded characters, token list and selected declarations.
def decodeTable (bytes : ByteArray) : Result CertifiedDecoded := do
  let (tokens, consumed) ← tokenize bytes
  decodeTokens bytes tokens consumed

-- Temporal O(n) for n source bytes: `tokenize` is linear and `scanSelected` visits the O(n) tokens
-- once against the one-name target. Spatial O(n): the decoded characters and token list.
def decodeImportedCoordCeiling (bytes : ByteArray) : Result ConstantDecl :=
  (tokenize bytes).bind fun result =>
    if result.2 = bytes.size then
      (scanSelected (result.1.length + 1) result.1 ["COORD_CEILING"] []).bind fun selected =>
        match selected.1 with
        | [] => .error "COORD_CEILING is missing or duplicated"
        | decl :: [] => .ok decl
        | _ :: _ :: _ => .error "COORD_CEILING is missing or duplicated"
    else .error "import tokenizer did not consume every source byte"

-- Temporal O(c) for c = |program.constants|: one `++ [coord]`; the certificate bounds c to the 8
-- selected constants, so O(1) on the decode path. Spatial O(c) for the appended list.
def appendImportedCoord (decoded : CertifiedDecoded) (coord : ConstantDecl) : CertifiedDecoded :=
  ⟨{ decoded.val with
      program := { decoded.val.program with
        constants := decoded.val.program.constants ++ [coord] } }, decoded.property⟩

-- Temporal O(n + m) for the table and std source byte lengths: two linear decodes plus one constant
-- comparison and append. Spatial O(n + m) for the two decoded token/declaration sets.
def decodeClosure (tableBytes stdBytes : ByteArray) : Result CertifiedDecoded := do
  let decoded ← decodeTable tableBytes
  let coord ← decodeImportedCoordCeiling stdBytes
  match coord.value with
  | .literal value =>
      match Nat.beq value 8796093022208 with
      | true => .ok (appendImportedCoord decoded coord)
      | false => .error "COORD_CEILING has the wrong value"
  | .name _ => .error "COORD_CEILING has the wrong value"
  | .call _ _ => .error "COORD_CEILING has the wrong value"
  | .memory _ => .error "COORD_CEILING has the wrong value"
  | .binary _ _ _ => .error "COORD_CEILING has the wrong value"
  | .conditional _ _ _ => .error "COORD_CEILING has the wrong value"

theorem decode_certified (source : ByteArray) (decoded : CertifiedDecoded)
    (_h : decodeTable source = .ok decoded) : decoded.val.certificate.Valid := decoded.property

end Iota.Decode
