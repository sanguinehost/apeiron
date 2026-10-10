namespace Iota

structure Span where
  start : Nat
  stop : Nat
  deriving Repr, BEq

inductive BinOp where
  | add | sub | mul
  | bitAnd | bitOr | bitXor
  | eq | ne | lt | le | gt | ge
  deriving Repr, BEq

inductive Expr where
  | literal (value : Nat)
  | name (value : String)
  | call (callee : String) (args : List Expr)
  | memory (address : Expr)
  | binary (op : BinOp) (left right : Expr)
  | conditional (thenValue condition elseValue : Expr)
  deriving Repr, BEq

inductive Stmt where
  | bind (name : String) (value : Expr)
  | store (address value : Expr)
  | return_ (value : Expr)
  | expression (value : Expr)
  | if_ (condition : Expr) (thenBody : List Stmt) (elseBody : Option (List Stmt))
  deriving Repr, BEq

structure ConstantDecl where
  name : String
  value : Expr
  span : Span
  deriving Repr, BEq

structure FunctionDecl where
  name : String
  parameters : List String
  body : List Stmt
  span : Span
  deriving Repr, BEq

structure Program where
  constants : List ConstantDecl
  functions : List FunctionDecl
  sourceBytes : Nat
  consumedBytes : Nat
  deriving Repr, BEq

namespace Syntax

-- Temporal O(n + c·d) for n Expr nodes, c emitted call names and depth d: one visit per node plus
-- the `++`/`flatMap` copies, each charging its left operand, so a left-leaning tree copies O(c) at
-- each of O(d) levels (O(n²) worst case; O(c log c) balanced). Spatial O(c + d): emitted list plus
-- one recursion frame per level, the branches binding their recursive result rather than tail-calling.
-- Operations counted: node visits, list elements copied, String conses; the recursion closes on the
-- structural subterms of `Expr`.
def Expr.calls : Expr → List String
  | .literal _ | .name _ => []
  | .call callee args => callee :: args.flatMap calls
  | .memory address => calls address
  | .binary _ left right => calls left ++ calls right
  | .conditional thenValue condition elseValue =>
      calls thenValue ++ calls condition ++ calls elseValue

-- As `Expr.calls`, with c the number of `.name` occurrences emitted: same temporal O(n + c·d) and
-- spatial O(c + d) accounting over the `Expr` tree.
def Expr.names : Expr → List String
  | .literal _ => []
  | .name value => [value]
  | .call _ args => args.flatMap names
  | .memory address => names address
  | .binary _ left right => names left ++ names right
  | .conditional thenValue condition elseValue =>
      names thenValue ++ names condition ++ names elseValue

mutual
  -- Temporal O(N + C) for N Stmt/Expr nodes and C emitted call names: each node is visited once, the
  -- `Expr.calls` concatenation terms are charged to their subexpressions, and `listCalls` copies only
  -- the current statement's names per step. Spatial O(C + d): emitted list plus one recursion frame
  -- per statement-nesting level. The mutual recursion closes on the structural subterms of `Stmt`
  -- and `List Stmt`.
  def Stmt.calls : Stmt → List String
    | .bind _ value | .return_ value | .expression value => Expr.calls value
    | .store address value => Expr.calls address ++ Expr.calls value
    | .if_ condition thenBody none =>
        Expr.calls condition ++ Stmt.listCalls thenBody
    | .if_ condition thenBody (some elseBody) =>
        Expr.calls condition ++ Stmt.listCalls thenBody ++ Stmt.listCalls elseBody

  -- Temporal O(m + C) for m statements and C emitted names: one step per statement, each appending
  -- its own `Stmt.calls` result (copying only that left operand). Spatial O(C + m) stack: `calls stmt
  -- ++ listCalls rest` binds the recursive result, so one frame per remaining statement.
  def Stmt.listCalls : List Stmt → List String
    | [] => []
    | stmt :: rest => calls stmt ++ listCalls rest
end

-- Temporal O(B + C) for B body Stmt/Expr nodes and C emitted call names: `List.flatMap Stmt.calls`
-- visits each body statement once and copies each statement's names once, plus the per-expression
-- concatenation terms. Spatial O(C + body depth): the emitted list plus the `Stmt.calls` stacks.
def FunctionDecl.calls (decl : FunctionDecl) : List String := decl.body.flatMap Stmt.calls

end Syntax
end Iota
