import Slang.Audit.Empty
import Slang.Derivation.Steps
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 4096

namespace Slang.Derivation.Statements

open Slang Slang.Decode

inductive Request where
  | block (fuel : Nat) (tokens : List Token)
  | stmt (fuel : Nat) (tokens : List Token)
  | expr (fuel : Nat) (tokens : List Token)

abbrev Answer : Request → Type
  | .block _ _ => Result (List Stmt × Token × List Token)
  | .stmt _ _ => Result (Stmt × List Token)
  | .expr _ _ => Result (Expr × List Token)

def evaluate : (q : Request) → Answer q
  | .block fuel tokens => Decode.parseBlockFuel fuel tokens
  | .stmt fuel tokens => Decode.parseStmtFuel fuel tokens
  | .expr fuel tokens => Decode.parseExprFuel fuel 0 tokens

/-- One statement leaf: parse an expression, expect a semicolon, then apply `mk`. -/
def exprSemiLayer (mk : Expr → Stmt) (ts : List Token) :
    Layer Request Answer (Result (Stmt × List Token)) :=
  .ask (.expr (ts.length * 4 + 16) ts) (fun result =>
    match result with
    | .error message => .done (.error message)
    | .ok (value, rest) =>
        match Decode.expectKind .semi (Decode.dropNewlines rest) with
        | .error message => .done (.error message)
        | .ok (_, rest2) => .done (.ok (mk value, rest2)))

/-- The `if` statement leaf, including the optional `else` block. -/
def ifLayer (fuel : Nat) (ts : List Token) :
    Layer Request Answer (Result (Stmt × List Token)) :=
  .ask (.expr (ts.length * 4 + 16) ts) (fun result =>
    match result with
    | .error message => .done (.error message)
    | .ok (condition, rest1) =>
        match Decode.expectKind .lbrace (Decode.dropNewlines rest1) with
        | .error message => .done (.error message)
        | .ok (_, rest2) =>
            .ask (.block fuel rest2) (fun result2 =>
              match result2 with
              | .error message => .done (.error message)
              | .ok (thenBody, _, rest3) =>
                  match Decode.dropNewlines rest3 with
                  | next :: tail =>
                      if next.kind = .ident "else" then
                        match Decode.expectKind .lbrace (Decode.dropNewlines tail) with
                        | .error message => .done (.error message)
                        | .ok (_, tail1) =>
                            .ask (.block fuel tail1) (fun result3 =>
                              match result3 with
                              | .error message => .done (.error message)
                              | .ok (elseBody, _, tail2) =>
                                  .done (.ok (.if_ condition thenBody (some elseBody), tail2)))
                      else .done (.ok (.if_ condition thenBody none, rest3))
                  | [] => .done (.ok (.if_ condition thenBody none, rest3))))

/-- The `memory[...] = ...` store leaf. -/
def memoryLayer (ts : List Token) :
    Layer Request Answer (Result (Stmt × List Token)) :=
  .ask (.expr (ts.length * 4 + 16) ts) (fun result =>
    match result with
    | .error message => .done (.error message)
    | .ok (address, rest1) =>
        match Decode.expectKind .rbracket rest1 with
        | .error message => .done (.error message)
        | .ok (_, rest2) =>
            match Decode.expectKind .assign (Decode.dropNewlines rest2) with
            | .error message => .done (.error message)
            | .ok (_, rest3) =>
                .ask (.expr (rest3.length * 4 + 16) rest3) (fun result2 =>
                  match result2 with
                  | .error message => .done (.error message)
                  | .ok (value, rest4) =>
                      match Decode.expectKind .semi (Decode.dropNewlines rest4) with
                      | .error message => .done (.error message)
                      | .ok (_, rest5) => .done (.ok (.store address value, rest5))))

def body : (q : Request) → Layer Request Answer (Answer q)
  | .block 0 _ => .done (.error "statement nesting limit reached")
  | .block (fuel + 1) tokens =>
      match Decode.dropNewlines tokens with
      | [] => .done (.error "unterminated function body")
      | token :: rest =>
          if token.kind = .rbrace then .done (.ok ([], token, rest)) else
            .ask (.stmt fuel (token :: rest)) (fun result =>
              match result with
              | .error message => .done (.error message)
              | .ok (stmt, rest1) =>
                  .ask (.block fuel rest1) (fun result2 =>
                    match result2 with
                    | .error message => .done (.error message)
                    | .ok (body, close, rest2) =>
                        .done (.ok (stmt :: body, close, rest2))))
  | .stmt 0 _ => .done (.error "statement nesting limit reached")
  | .stmt (fuel + 1) tokens =>
      match Decode.dropNewlines tokens with
      | token :: rest =>
          match token.kind with
          | .ident name =>
              if name = "return" then exprSemiLayer Stmt.return_ rest
              else if name = "if" then ifLayer fuel rest
              else if name = "memory" then
                match rest with
                | next :: tail =>
                    if next.kind = .lbracket then memoryLayer tail
                    else exprSemiLayer Stmt.expression (token :: next :: tail)
                | [] => exprSemiLayer Stmt.expression [token]
              else
                match rest with
                | next :: tail =>
                    if next.kind = .assign then exprSemiLayer (Stmt.bind name) tail
                    else exprSemiLayer Stmt.expression (token :: next :: tail)
                | [] => exprSemiLayer Stmt.expression [token]
          | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
              .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
              exprSemiLayer Stmt.expression (token :: rest)
      | [] => exprSemiLayer Stmt.expression []
  | .expr fuel tokens => .done (Decode.parseExprFuel fuel 0 tokens)

abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

private theorem bind_ok {ε α β : Type} (value : α) (next : α → Except ε β) :
    (Except.ok value >>= next) = next value := rfl

private theorem bind_error {ε α β : Type} (message : ε) (next : α → Except ε β) :
    (Except.error message >>= next) = Except.error message := rfl

private theorem expr_semi_fold (mk : Expr → Stmt) (fuel : Nat) (tokens ts : List Token) :
    (fold evaluate (exprSemiLayer mk ts) : Answer (.stmt fuel tokens))
    = (do
        let value ← Decode.parseExprFuel (ts.length * 4 + 16) 0 ts
        let rest ← Decode.expectKind .semi (Decode.dropNewlines value.snd)
        Except.ok (mk value.fst, rest.snd)) := by
  unfold Slang.Derivation.Statements.exprSemiLayer
  rw [Slang.Derivation.fold_ask]
  rw [show Slang.Derivation.Statements.evaluate
        (Slang.Derivation.Statements.Request.expr (ts.length * 4 + 16) ts)
      = Decode.parseExprFuel (ts.length * 4 + 16) 0 ts from rfl]
  cases hx : Decode.parseExprFuel (ts.length * 4 + 16) 0 ts with
  | error message =>
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      rfl
  | ok pair =>
      cases pair with
      | mk value rest =>
          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
          cases he : Decode.expectKind .semi (Decode.dropNewlines rest) with
          | error message => rfl
          | ok pair2 =>
              cases pair2 with
              | mk _ rest2 => rfl

private theorem if_fold (reqFuel : Nat) (reqTokens : List Token) (fuel : Nat) (ts : List Token) :
    (fold evaluate (ifLayer fuel ts) : Answer (.stmt reqFuel reqTokens))
    = (do
        let condition ← Decode.parseExprFuel (ts.length * 4 + 16) 0 ts
        let rest1 ← Decode.expectKind .lbrace (Decode.dropNewlines condition.snd)
        let thenBlock ← Decode.parseBlockFuel fuel rest1.snd
        (match Decode.dropNewlines thenBlock.2.snd with
          | next :: tail =>
              (if next.kind = TokenKind.ident "else" then
                (do
                  let rest2 ← Decode.expectKind .lbrace (Decode.dropNewlines tail)
                  let elseBlock ← Decode.parseBlockFuel fuel rest2.snd
                  Except.ok (Stmt.if_ condition.fst thenBlock.fst (some elseBlock.fst), elseBlock.2.snd))
              else Except.ok (Stmt.if_ condition.fst thenBlock.fst none, thenBlock.2.snd))
          | [] => Except.ok (Stmt.if_ condition.fst thenBlock.fst none, thenBlock.2.snd))) := by
  unfold Slang.Derivation.Statements.ifLayer
  rw [Slang.Derivation.fold_ask]
  rw [show Slang.Derivation.Statements.evaluate
        (Slang.Derivation.Statements.Request.expr (ts.length * 4 + 16) ts)
      = Decode.parseExprFuel (ts.length * 4 + 16) 0 ts from rfl]
  cases hx : Decode.parseExprFuel (ts.length * 4 + 16) 0 ts with
  | error message =>
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      rfl
  | ok pair =>
      cases pair with
      | mk condition rest1 =>
          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
          cases he : Decode.expectKind .lbrace (Decode.dropNewlines rest1) with
          | error message => rfl
          | ok pair2 =>
              cases pair2 with
              | mk _ rest2 =>
                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                  rw [Slang.Derivation.fold_ask]
                  rw [show Slang.Derivation.Statements.evaluate
                        (Slang.Derivation.Statements.Request.block fuel rest2)
                      = Decode.parseBlockFuel fuel rest2 from rfl]
                  cases hb : Decode.parseBlockFuel fuel rest2 with
                  | error message =>
                      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                      rfl
                  | ok triple =>
                      cases triple with
                      | mk thenBody rest34 =>
                          cases rest34 with
                          | mk close rest3 =>
                              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                              cases hd2 : Decode.dropNewlines rest3 with
                              | nil =>
                                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                  rfl
                              | cons next tail =>
                                  dsimp only []
                                  by_cases helse : next.kind = TokenKind.ident "else"
                                  · conv => lhs; rw [if_pos helse]
                                    conv => rhs; rw [if_pos helse]
                                    cases he2 : Decode.expectKind .lbrace (Decode.dropNewlines tail) with
                                    | error message => rfl
                                    | ok pair3 =>
                                        cases pair3 with
                                        | mk _ tail1 =>
                                            dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                            rw [Slang.Derivation.fold_ask]
                                            rw [show Slang.Derivation.Statements.evaluate
                                                  (Slang.Derivation.Statements.Request.block fuel tail1)
                                                = Decode.parseBlockFuel fuel tail1 from rfl]
                                            cases hb2 : Decode.parseBlockFuel fuel tail1 with
                                            | error message =>
                                                dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                                rfl
                                            | ok triple2 =>
                                                cases triple2 with
                                                | mk elseBody rest45 =>
                                                    cases rest45 with
                                                    | mk _ tail2 =>
                                                        dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                                        rfl
                                  · conv => lhs; rw [if_neg helse]
                                    conv => rhs; rw [if_neg helse]
                                    rfl

private theorem memory_fold (reqFuel : Nat) (reqTokens : List Token) (ts : List Token) :
    (fold evaluate (memoryLayer ts) : Answer (.stmt reqFuel reqTokens))
    = (do
        let address ← Decode.parseExprFuel (ts.length * 4 + 16) 0 ts
        let rest1 ← Decode.expectKind .rbracket address.snd
        let rest2 ← Decode.expectKind .assign (Decode.dropNewlines rest1.snd)
        let value ← Decode.parseExprFuel (rest2.snd.length * 4 + 16) 0 rest2.snd
        let rest3 ← Decode.expectKind .semi (Decode.dropNewlines value.snd)
        Except.ok (Stmt.store address.fst value.fst, rest3.snd)) := by
  unfold Slang.Derivation.Statements.memoryLayer
  rw [Slang.Derivation.fold_ask]
  rw [show Slang.Derivation.Statements.evaluate
        (Slang.Derivation.Statements.Request.expr (ts.length * 4 + 16) ts)
      = Decode.parseExprFuel (ts.length * 4 + 16) 0 ts from rfl]
  cases hx : Decode.parseExprFuel (ts.length * 4 + 16) 0 ts with
  | error message =>
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      rfl
  | ok pair =>
      cases pair with
      | mk address rest1 =>
          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
          cases he : Decode.expectKind .rbracket rest1 with
          | error message => rfl
          | ok pair2 =>
              cases pair2 with
              | mk _ rest2 =>
                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                  cases he2 : Decode.expectKind .assign (Decode.dropNewlines rest2) with
                  | error message => rfl
                  | ok pair3 =>
                      cases pair3 with
                      | mk _ rest3 =>
                          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          rw [Slang.Derivation.fold_ask]
                          rw [show Slang.Derivation.Statements.evaluate
                                (Slang.Derivation.Statements.Request.expr (rest3.length * 4 + 16) rest3)
                              = Decode.parseExprFuel (rest3.length * 4 + 16) 0 rest3 from rfl]
                          cases hx2 : Decode.parseExprFuel (rest3.length * 4 + 16) 0 rest3 with
                          | error message =>
                              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                              rfl
                          | ok pair4 =>
                              cases pair4 with
                              | mk value rest4 =>
                                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                  cases he3 : Decode.expectKind .semi (Decode.dropNewlines rest4) with
                                  | error message => rfl
                                  | ok pair5 =>
                                      cases pair5 with
                                      | mk _ rest5 => rfl

theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .block 0 _ => rfl
  | .block (fuel + 1) tokens => by
      rw [show Slang.Derivation.Statements.evaluate
            (Slang.Derivation.Statements.Request.block (fuel + 1) tokens)
          = Decode.parseBlockFuel (fuel + 1) tokens from rfl]
      dsimp only [Slang.Derivation.Statements.body]
      rw [Decode.parseBlockFuel]
      dsimp only []
      cases hd : Decode.dropNewlines tokens with
      | nil => rfl
      | cons token rest =>
          dsimp only []
          by_cases hb : token.kind = .rbrace
          · conv => lhs; rw [if_pos hb]
            conv => rhs; rw [if_pos hb]
            rfl
          · conv => lhs; rw [if_neg hb]
            conv => rhs; rw [if_neg hb]
            rw [Slang.Derivation.fold_ask]
            cases hs : Decode.parseStmtFuel fuel (token :: rest) with
            | error message =>
                rw [show Slang.Derivation.Statements.evaluate
                      (Slang.Derivation.Statements.Request.stmt fuel (token :: rest))
                    = Decode.parseStmtFuel fuel (token :: rest) from rfl]
                rw [hs]
                rfl
            | ok pair =>
                cases pair with
                | mk stmt rest1 =>
                    rw [show Slang.Derivation.Statements.evaluate
                          (Slang.Derivation.Statements.Request.stmt fuel (token :: rest))
                        = Decode.parseStmtFuel fuel (token :: rest) from rfl]
                    rw [hs]
                    dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                    rw [Slang.Derivation.fold_ask]
                    rw [show Slang.Derivation.Statements.evaluate
                          (Slang.Derivation.Statements.Request.block fuel rest1)
                        = Decode.parseBlockFuel fuel rest1 from rfl]
                    cases hb2 : Decode.parseBlockFuel fuel rest1 with
                    | error message => rfl
                    | ok triple =>
                        cases triple with
                        | mk bodyStmts rest23 =>
                            cases rest23 with
                            | mk close rest2 => rfl
  | .stmt 0 _ => rfl
  | .stmt (fuel + 1) tokens => by
      rw [show Slang.Derivation.Statements.evaluate
            (Slang.Derivation.Statements.Request.stmt (fuel + 1) tokens)
          = Decode.parseStmtFuel (fuel + 1) tokens from rfl]
      dsimp only [Slang.Derivation.Statements.body]
      rw [Decode.parseStmtFuel]
      dsimp only []
      cases hd : Decode.dropNewlines tokens with
      | nil =>
          dsimp only []
          rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens ([] : List Token)]
          rfl
      | cons token rest =>
          dsimp only []
          cases hk : token.kind
          case ident name =>
            dsimp only []
            by_cases hret : name = "return"
            · conv => lhs; rw [if_pos hret]
              conv => rhs; rw [if_pos hret]
              rw [Slang.Derivation.Statements.expr_semi_fold Stmt.return_ (fuel + 1) tokens rest]
              rfl
            · conv => lhs; rw [if_neg hret]
              conv => rhs; rw [if_neg hret]
              by_cases hif : name = "if"
              · conv => lhs; rw [if_pos hif]
                conv => rhs; rw [if_pos hif]
                rw [Slang.Derivation.Statements.if_fold (fuel + 1) tokens fuel rest]
                rfl
              · conv => lhs; rw [if_neg hif]
                conv => rhs; rw [if_neg hif]
                by_cases hmem : name = "memory"
                · conv => lhs; rw [if_pos hmem]
                  conv => rhs; rw [if_pos hmem]
                  cases rest with
                  | nil =>
                      rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens [token]]
                      rfl
                  | cons next tail =>
                      dsimp only []
                      by_cases hbr : next.kind = TokenKind.lbracket
                      · conv => lhs; rw [if_pos hbr]
                        conv => rhs; rw [if_pos hbr]
                        rw [Slang.Derivation.Statements.memory_fold (fuel + 1) tokens tail]
                        rfl
                      · conv => lhs; rw [if_neg hbr]
                        conv => rhs; rw [if_neg hbr]
                        rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens (token :: next :: tail)]
                        rfl
                · conv => lhs; rw [if_neg hmem]
                  conv => rhs; rw [if_neg hmem]
                  cases rest with
                  | nil =>
                      rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens [token]]
                      rfl
                  | cons next tail =>
                      dsimp only []
                      by_cases has : next.kind = TokenKind.assign
                      · conv => lhs; rw [if_pos has]
                        conv => rhs; rw [if_pos has]
                        rw [Slang.Derivation.Statements.expr_semi_fold (Stmt.bind name) (fuel + 1) tokens tail]
                        rfl
                      · conv => lhs; rw [if_neg has]
                        conv => rhs; rw [if_neg has]
                        rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens (token :: next :: tail)]
                        rfl
          all_goals
            (dsimp only []
             rw [Slang.Derivation.Statements.expr_semi_fold Stmt.expression (fuel + 1) tokens (token :: rest)]
             rfl)
  | .expr fuel tokens => rfl

theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

syntax "derives_done" : tactic
macro_rules
  | `(tactic| derives_done) =>
    `(tactic| (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]); (try dsimp only []); exact Slang.Derivation.Derives.done _)

mutual
  /-- Every block request has a checked derivation of the real parser's answer. -/
  theorem block_run : ∀ (fuel : Nat) (tokens : List Token),
      Run (.block fuel tokens) (Decode.parseBlockFuel fuel tokens)
    | 0, _ => Slang.Derivation.Derives.done _
    | fuel + 1, tokens => by
        unfold Slang.Derivation.Statements.Run Slang.Derivation.Statements.body
          Decode.parseBlockFuel
        dsimp only []
        cases hd : Decode.dropNewlines tokens with
        | nil => derives_done
        | cons token rest =>
            dsimp only []
            by_cases hb : token.kind = .rbrace
            · conv => arg 2; rw [if_pos hb]
              conv => arg 3; rw [if_pos hb]
              derives_done
            · conv => arg 2; rw [if_neg hb]
              conv => arg 3; rw [if_neg hb]
              refine Slang.Derivation.Derives.ask
                (body := Slang.Derivation.Statements.body)
                (request := .stmt fuel (token :: rest))
                (answer := Decode.parseStmtFuel fuel (token :: rest))
                (stmt_run fuel (token :: rest)) ?_
              cases hs : Decode.parseStmtFuel fuel (token :: rest) with
              | error message =>
                  dsimp only []
                  derives_done
              | ok pair =>
                  cases pair with
                  | mk stmt rest1 =>
                      dsimp only []
                      refine Slang.Derivation.Derives.ask
                        (body := Slang.Derivation.Statements.body)
                        (request := .block fuel rest1)
                        (answer := Decode.parseBlockFuel fuel rest1)
                        (block_run fuel rest1) ?_
                      cases hb2 : Decode.parseBlockFuel fuel rest1 with
                      | error message =>
                          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          rw [hb2]
                          derives_done
                      | ok triple =>
                          cases triple with
                          | mk bodyStmts rest23 =>
                              cases rest23 with
                              | mk close rest2 =>
                                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                                  rw [hb2]
                                  derives_done

  /-- Every statement request has a checked derivation of the real parser's answer. -/
  theorem stmt_run : ∀ (fuel : Nat) (tokens : List Token),
      Run (.stmt fuel tokens) (Decode.parseStmtFuel fuel tokens)
    | 0, _ => Slang.Derivation.Derives.done _
    | fuel + 1, tokens => by
        unfold Slang.Derivation.Statements.Run Slang.Derivation.Statements.body
          Decode.parseStmtFuel
        dsimp only []
        cases hd : Decode.dropNewlines tokens with
        | nil =>
            exact Slang.Derivation.Derives.ask
              (body := Slang.Derivation.Statements.body)
              (request := .expr ((([] : List Token)).length * 4 + 16) ([] : List Token))
              (answer := Decode.parseExprFuel ((([] : List Token)).length * 4 + 16) 0 ([] : List Token))
              (Slang.Derivation.Derives.done _) (by
                conv => arg 3; change (do let value ← Decode.parseExprFuel ((([] : List Token)).length * 4 + 16) 0 ([] : List Token); let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                cases hx : Decode.parseExprFuel ((([] : List Token)).length * 4 + 16) 0 ([] : List Token) with
                | error message => derives_done
                | ok pair =>
                    cases pair with
                    | mk value rest1 =>
                        dsimp only []
                        (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                        | error message => derives_done
                        | ok pair2 =>
                            cases pair2 with
                            | mk _ rest2 => derives_done)
        | cons token rest =>
            dsimp only []
            cases hk : token.kind
            case ident name =>
              dsimp only []
              by_cases hret : name = "return"
              · conv => arg 2; rw [if_pos hret]
                conv => arg 3; rw [if_pos hret]
                exact Slang.Derivation.Derives.ask
                  (body := Slang.Derivation.Statements.body)
                  (request := .expr (rest.length * 4 + 16) rest)
                  (answer := Decode.parseExprFuel (rest.length * 4 + 16) 0 rest)
                  (Slang.Derivation.Derives.done _) (by
                    conv => arg 3; change (do let value ← Decode.parseExprFuel (rest.length * 4 + 16) 0 rest; let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.return_ value.fst, rest1.snd))
                    cases hx : Decode.parseExprFuel (rest.length * 4 + 16) 0 rest with
                    | error message => derives_done
                    | ok pair =>
                        cases pair with
                        | mk value rest1 =>
                            dsimp only []
                            (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                            | error message => derives_done
                            | ok pair2 =>
                                cases pair2 with
                                | mk _ rest2 => derives_done)
              · conv => arg 2; rw [if_neg hret]
                conv => arg 3; rw [if_neg hret]
                by_cases hif : name = "if"
                · conv => arg 2; rw [if_pos hif]
                  conv => arg 3; rw [if_pos hif]
                  exact Slang.Derivation.Derives.ask
                    (body := Slang.Derivation.Statements.body)
                    (request := .expr (rest.length * 4 + 16) rest)
                    (answer := Decode.parseExprFuel (rest.length * 4 + 16) 0 rest)
                    (Slang.Derivation.Derives.done _) (by
                      conv => arg 3; change (do
                        let condition ← Decode.parseExprFuel (rest.length * 4 + 16) 0 rest
                        let rest1 ← Decode.expectKind TokenKind.lbrace (Decode.dropNewlines condition.snd)
                        let thenBlock ← Decode.parseBlockFuel fuel rest1.snd
                        (match Decode.dropNewlines thenBlock.2.snd with
                          | next :: tail =>
                              (if next.kind = TokenKind.ident "else" then
                                (do
                                  let rest2 ← Decode.expectKind TokenKind.lbrace (Decode.dropNewlines tail)
                                  let elseBlock ← Decode.parseBlockFuel fuel rest2.snd
                                  Except.ok (Stmt.if_ condition.fst thenBlock.fst (some elseBlock.fst), elseBlock.2.snd))
                              else Except.ok (Stmt.if_ condition.fst thenBlock.fst none, thenBlock.2.snd))
                          | [] => Except.ok (Stmt.if_ condition.fst thenBlock.fst none, thenBlock.2.snd)))
                      cases hx : Decode.parseExprFuel (rest.length * 4 + 16) 0 rest with
                      | error message => derives_done
                      | ok pair =>
                          cases pair with
                          | mk condition rest1 =>
                              dsimp only []
                              (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .lbrace (Decode.dropNewlines rest1) with
                              | error message => derives_done
                              | ok pair2 =>
                                  cases pair2 with
                                  | mk _ rest2 =>
                                      exact Slang.Derivation.Derives.ask
                                        (body := Slang.Derivation.Statements.body)
                                        (request := .block fuel rest2)
                                        (answer := Decode.parseBlockFuel fuel rest2)
                                        (block_run fuel rest2) (by
                                          (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases hb : Decode.parseBlockFuel fuel rest2 with
                                          | error message => derives_done
                                          | ok triple =>
                                              cases triple with
                                              | mk thenBody rest23 =>
                                                cases rest23 with
                                                | mk close rest3 =>
                                                  dsimp only []
                                                  (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases hd2 : Decode.dropNewlines rest3 with
                                                  | nil => derives_done
                                                  | cons next tail =>
                                                      dsimp only []
                                                      by_cases helse : next.kind = TokenKind.ident "else"
                                                      · conv => arg 2; rw [if_pos helse]
                                                        conv => arg 3; rw [if_pos helse]
                                                        (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he2 : Decode.expectKind .lbrace (Decode.dropNewlines tail) with
                                                        | error message => derives_done
                                                        | ok pair3 =>
                                                            cases pair3 with
                                                            | mk _ tail1 =>
                                                                exact Slang.Derivation.Derives.ask
                                                                  (body := Slang.Derivation.Statements.body)
                                                                  (request := .block fuel tail1)
                                                                  (answer := Decode.parseBlockFuel fuel tail1)
                                                                  (block_run fuel tail1) (by
                                                                    (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases hb2 : Decode.parseBlockFuel fuel tail1 with
                                                                    | error message => derives_done
                                                                    | ok triple2 =>
                                                                        cases triple2 with
                                                                        | mk elseBody rest67 =>
                                                                            cases rest67 with
                                                                            | mk close2 tail2 =>
                                                                                derives_done)
                                                      · conv => arg 2; rw [if_neg helse]
                                                        conv => arg 3; rw [if_neg helse]
                                                        derives_done))
                · conv => arg 2; rw [if_neg hif]
                  conv => arg 3; rw [if_neg hif]
                  by_cases hmem : name = "memory"
                  · conv => arg 2; rw [if_pos hmem]
                    conv => arg 3; rw [if_pos hmem]
                    cases rest with
                    | nil =>
                        exact Slang.Derivation.Derives.ask
                          (body := Slang.Derivation.Statements.body)
                          (request := .expr ([token].length * 4 + 16) [token])
                          (answer := Decode.parseExprFuel ([token].length * 4 + 16) 0 [token])
                          (Slang.Derivation.Derives.done _) (by
                            conv => arg 3; change (do let value ← Decode.parseExprFuel ([token].length * 4 + 16) 0 [token]; let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                            cases hx : Decode.parseExprFuel ([token].length * 4 + 16) 0 [token] with
                            | error message => derives_done
                            | ok pair =>
                                cases pair with
                                | mk value rest1 =>
                                    dsimp only []
                                    (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                                    | error message => derives_done
                                    | ok pair2 =>
                                        cases pair2 with
                                        | mk _ rest2 => derives_done)
                    | cons next tail =>
                        dsimp only []
                        by_cases hbr : next.kind = TokenKind.lbracket
                        · conv => arg 2; rw [if_pos hbr]
                          conv => arg 3; rw [if_pos hbr]
                          exact Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Statements.body)
                            (request := .expr (tail.length * 4 + 16) tail)
                            (answer := Decode.parseExprFuel (tail.length * 4 + 16) 0 tail)
                            (Slang.Derivation.Derives.done _) (by
                              conv => arg 3; change (do
                                let address ← Decode.parseExprFuel (tail.length * 4 + 16) 0 tail
                                let rest1 ← Decode.expectKind TokenKind.rbracket address.snd
                                let rest2 ← Decode.expectKind TokenKind.assign (Decode.dropNewlines rest1.snd)
                                let value ← Decode.parseExprFuel (rest2.snd.length * 4 + 16) 0 rest2.snd
                                let rest3 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd)
                                Except.ok (Stmt.store address.fst value.fst, rest3.snd))
                              cases hx : Decode.parseExprFuel (tail.length * 4 + 16) 0 tail with
                              | error message => derives_done
                              | ok pair =>
                                  cases pair with
                                  | mk address rest1 =>
                                      dsimp only []
                                      (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .rbracket rest1 with
                                      | error message => derives_done
                                      | ok pair2 =>
                                          cases pair2 with
                                          | mk _ rest2 =>
                                              dsimp only []
                                              (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he2 : Decode.expectKind .assign (Decode.dropNewlines rest2) with
                                              | error message => derives_done
                                              | ok pair3 =>
                                                  cases pair3 with
                                                  | mk _ rest3 =>
                                                      exact Slang.Derivation.Derives.ask
                                                        (body := Slang.Derivation.Statements.body)
                                                        (request := .expr (rest3.length * 4 + 16) rest3)
                                                        (answer := Decode.parseExprFuel (rest3.length * 4 + 16) 0 rest3)
                                                        (Slang.Derivation.Derives.done _) (by
                                                          (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases hx2 : Decode.parseExprFuel (rest3.length * 4 + 16) 0 rest3 with
                                                          | error message => derives_done
                                                          | ok pair4 =>
                                                              cases pair4 with
                                                              | mk value rest4 =>
                                                                  dsimp only []
                                                                  (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he3 : Decode.expectKind .semi (Decode.dropNewlines rest4) with
                                                                  | error message => derives_done
                                                                  | ok pair5 =>
                                                                      cases pair5 with
                                                                      | mk _ rest5 => derives_done))
                        · conv => arg 2; rw [if_neg hbr]
                          conv => arg 3; rw [if_neg hbr]
                          exact Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Statements.body)
                            (request := .expr ((token :: next :: tail).length * 4 + 16) (token :: next :: tail))
                            (answer := Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail))
                            (Slang.Derivation.Derives.done _) (by
                              conv => arg 3; change (do let value ← Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail); let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                              cases hx : Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail) with
                              | error message => derives_done
                              | ok pair =>
                                  cases pair with
                                  | mk value rest1 =>
                                      dsimp only []
                                      (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                                      | error message => derives_done
                                      | ok pair2 =>
                                          cases pair2 with
                                          | mk _ rest2 => derives_done)
                  · conv => arg 2; rw [if_neg hmem]
                    conv => arg 3; rw [if_neg hmem]
                    cases rest with
                    | nil =>
                        exact Slang.Derivation.Derives.ask
                          (body := Slang.Derivation.Statements.body)
                          (request := .expr ([token].length * 4 + 16) [token])
                          (answer := Decode.parseExprFuel ([token].length * 4 + 16) 0 [token])
                          (Slang.Derivation.Derives.done _) (by
                            conv => arg 3; change (do let value ← Decode.parseExprFuel ([token].length * 4 + 16) 0 [token]; let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                            cases hx : Decode.parseExprFuel ([token].length * 4 + 16) 0 [token] with
                            | error message => derives_done
                            | ok pair =>
                                cases pair with
                                | mk value rest1 =>
                                    dsimp only []
                                    (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                                    | error message => derives_done
                                    | ok pair2 =>
                                        cases pair2 with
                                        | mk _ rest2 => derives_done)
                    | cons next tail =>
                        dsimp only []
                        by_cases has : next.kind = TokenKind.assign
                        · conv => arg 2; rw [if_pos has]
                          conv => arg 3; rw [if_pos has]
                          exact Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Statements.body)
                            (request := .expr (tail.length * 4 + 16) tail)
                            (answer := Decode.parseExprFuel (tail.length * 4 + 16) 0 tail)
                            (Slang.Derivation.Derives.done _) (by
                              conv => arg 3; change (do let value ← Decode.parseExprFuel (tail.length * 4 + 16) 0 tail; let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.bind name value.fst, rest1.snd))
                              cases hx : Decode.parseExprFuel (tail.length * 4 + 16) 0 tail with
                              | error message => derives_done
                              | ok pair =>
                                  cases pair with
                                  | mk value rest1 =>
                                      dsimp only []
                                      (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                                      | error message => derives_done
                                      | ok pair2 =>
                                          cases pair2 with
                                          | mk _ rest2 => derives_done)
                        · conv => arg 2; rw [if_neg has]
                          conv => arg 3; rw [if_neg has]
                          exact Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Statements.body)
                            (request := .expr ((token :: next :: tail).length * 4 + 16) (token :: next :: tail))
                            (answer := Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail))
                            (Slang.Derivation.Derives.done _) (by
                              conv => arg 3; change (do let value ← Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail); let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                              cases hx : Decode.parseExprFuel ((token :: next :: tail).length * 4 + 16) 0 (token :: next :: tail) with
                              | error message => derives_done
                              | ok pair =>
                                  cases pair with
                                  | mk value rest1 =>
                                      dsimp only []
                                      (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                                      | error message => derives_done
                                      | ok pair2 =>
                                          cases pair2 with
                                          | mk _ rest2 => derives_done)

            all_goals
              (dsimp only []
               exact Slang.Derivation.Derives.ask
                 (body := Slang.Derivation.Statements.body)
                 (request := .expr ((token :: rest).length * 4 + 16) (token :: rest))
                 (answer := Decode.parseExprFuel ((token :: rest).length * 4 + 16) 0 (token :: rest))
                 (Slang.Derivation.Derives.done _) (by
                   conv => arg 3; change (do let value ← Decode.parseExprFuel ((token :: rest).length * 4 + 16) 0 (token :: rest); let rest1 ← Decode.expectKind TokenKind.semi (Decode.dropNewlines value.snd); Except.ok (Stmt.expression value.fst, rest1.snd))
                   cases hx : Decode.parseExprFuel ((token :: rest).length * 4 + 16) 0 (token :: rest) with
                   | error message => derives_done
                   | ok pair =>
                       cases pair with
                       | mk value rest1 =>
                           dsimp only []
                           (try dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]) ; cases he : Decode.expectKind .semi (Decode.dropNewlines rest1) with
                           | error message => derives_done
                           | ok pair2 =>
                               cases pair2 with
                               | mk _ rest2 => derives_done))
end

end Slang.Derivation.Statements
