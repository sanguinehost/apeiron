namespace Slang

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

def Expr.calls : Expr → List String
  | .literal _ | .name _ => []
  | .call callee args => callee :: args.flatMap calls
  | .memory address => calls address
  | .binary _ left right => calls left ++ calls right
  | .conditional thenValue condition elseValue =>
      calls thenValue ++ calls condition ++ calls elseValue

def Expr.names : Expr → List String
  | .literal _ => []
  | .name value => [value]
  | .call _ args => args.flatMap names
  | .memory address => names address
  | .binary _ left right => names left ++ names right
  | .conditional thenValue condition elseValue =>
      names thenValue ++ names condition ++ names elseValue

mutual
  def Stmt.calls : Stmt → List String
    | .bind _ value | .return_ value | .expression value => Expr.calls value
    | .store address value => Expr.calls address ++ Expr.calls value
    | .if_ condition thenBody none =>
        Expr.calls condition ++ Stmt.listCalls thenBody
    | .if_ condition thenBody (some elseBody) =>
        Expr.calls condition ++ Stmt.listCalls thenBody ++ Stmt.listCalls elseBody

  def Stmt.listCalls : List Stmt → List String
    | [] => []
    | stmt :: rest => calls stmt ++ listCalls rest
end

def FunctionDecl.calls (decl : FunctionDecl) : List String := decl.body.flatMap Stmt.calls

end Syntax
end Slang
