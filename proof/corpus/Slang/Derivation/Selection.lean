import Slang.Derivation.Steps
import Slang.Derivation.Grammar
import Slang.Derivation.Statements
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 4096

namespace Slang.Derivation.Selection

open Slang Slang.Decode

/-- Requests name the recursive work of the declaration scanner instead of evaluating it. -/
inductive Request where
  | parameters (fuel : Nat) (tokens : List Token)
  | line (tokens : List Token)
  | function (name : String) (start : Nat) (tokens : List Token)
  | scan (fuel : Nat) (tokens : List Token) (constants functions : List String)
  | expr (fuel minPrec : Nat) (tokens : List Token)
  | block (fuel : Nat) (tokens : List Token)

/-- Answer type per request. -/
abbrev Answer : Request → Type
  | .parameters _ _ => Result (List String × List Token)
  | .line _ => List Token × List Token
  | .function _ _ _ => Result (FunctionDecl × List Token)
  | .scan _ _ _ _ => Result (List ConstantDecl × List FunctionDecl)
  | .expr _ _ _ => Result (Expr × List Token)
  | .block _ _ => Result (List Stmt × Token × List Token)

/-- The real declaration scanner, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .parameters fuel tokens => Decode.parseParameters fuel tokens
  | .line tokens => Decode.takeLine tokens
  | .function name start tokens => Decode.parseFunction name start tokens
  | .scan fuel tokens constants functions => Decode.scanSelected fuel tokens constants functions
  | .expr fuel minPrec tokens => Decode.parseExprFuel fuel minPrec tokens
  | .block fuel tokens => Decode.parseBlockFuel fuel tokens

/-- The span stop of a scanned constant line, with the header token as fallback. -/
def lineStop (line : List Token) (fallback : Nat) : Nat :=
  line.getLast?.map (fun token => token.span.stop) |>.getD fallback

/-- One unfolding of the scanner, exposing recursive calls as requests.

The `expr` and `block` requests are external-call leaves: their `body` is
`.done` of the real `Decode.parseExprFuel` / `Decode.parseBlockFuel` call. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .parameters 0 _ => .done (.error "parameter limit reached")
  | .parameters (_ + 1) [] => .done (.error "expected name at end of input")
  | .parameters (fuel + 1) (token :: rest) =>
      if token.kind = .rparen then .done (.ok ([], rest)) else
      match token.kind with
      | .ident name =>
          match rest with
          | [] => .done (.error "unterminated parameter list")
          | next :: tail =>
              if next.kind = .comma then
                .ask (.parameters fuel tail) (fun result =>
                  match result with
                  | .error message => .done (.error message)
                  | .ok (names, tail2) => .done (.ok (name :: names, tail2)))
              else if next.kind = .rparen then .done (.ok ([name], tail))
              else .done (.error "expected ',' or ')' in parameters")
      | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
          .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
          .done (.error "expected name")
  | .line [] => .done ([], [])
  | .line (token :: rest) =>
      if token.kind = .newline then .done ([], token :: rest) else
        .ask (.line rest) (fun result =>
          match result with
          | (line, tail) => .done (token :: line, tail))
  | .function name start tokens =>
      match Decode.expectKind .lparen tokens with
      | .error message => .done (.error message)
      | .ok (_, tokens1) =>
          .ask (.parameters (tokens1.length + 1) tokens1) (fun result =>
            match result with
            | .error message => .done (.error message)
            | .ok (parameters, tokens2) =>
                match Decode.expectKind .lbrace (Decode.dropNewlines tokens2) with
                | .error message => .done (.error message)
                | .ok (_, tokens3) =>
                    .ask (.block (tokens3.length + 16) tokens3) (fun result3 =>
                      match result3 with
                      | .error message => .done (.error message)
                      | .ok (body, close, tokens4) =>
                          .done (.ok ({ name, parameters, body,
                                        span := ⟨start, close.span.stop⟩ }, tokens4))))
  | .scan 0 _ _ _ => .done (.error "declaration scan limit reached")
  | .scan _ [] _ _ => .done (.ok ([], []))
  | .scan (fuel + 1) (token :: rest) constants functions =>
      match token.kind with
      | .ident name =>
          match rest with
          | [] => .ask (.scan fuel [] constants functions) (fun result => .done result)
          | next :: tail =>
              if next.kind = .assign then
                if constants.contains name then
                  .ask (.line tail) (fun lineResult =>
                    match lineResult with
                    | (line, afterLine) =>
                        .ask (.expr (line.length * 4 + 16) 0 line) (fun exprResult =>
                          match exprResult with
                          | .error message => .done (.error message)
                          | .ok (value, leftover) =>
                              if Decode.dropNewlines leftover != [] then
                                .done (.error "trailing constant tokens")
                              else
                                .ask (.scan fuel afterLine (constants.erase name) functions)
                                  (fun scanResult =>
                                    match scanResult with
                                    | .error message => .done (.error message)
                                    | .ok (decls, funcs) =>
                                        .done (.ok ({ name, value,
                                                      span := ⟨token.span.start,
                                                        lineStop line token.span.stop⟩ } :: decls,
                                                    funcs)))))
                else if Decode.selectedConstantNames.contains name then
                  .done (.error "duplicate selected constant")
                else .ask (.scan fuel rest constants functions) (fun result => .done result)
              else if next.kind = .lparen then
                if functions.contains name then
                  .ask (.function name token.span.start
                      ({ kind := .lparen, span := ⟨token.span.stop, token.span.stop + 1⟩ } :: tail))
                    (fun result =>
                      match result with
                      | .error message => .done (.error message)
                      | .ok (decl, afterDecl) =>
                          .ask (.scan fuel afterDecl constants (functions.erase name))
                            (fun scanResult =>
                              match scanResult with
                              | .error message => .done (.error message)
                              | .ok (decls, funcs) => .done (.ok (decls, decl :: funcs))))
                else .ask (.scan fuel rest constants functions) (fun result => .done result)
              else .ask (.scan fuel rest constants functions) (fun result => .done result)
      | .numeral _ | .lparen | .rparen | .lbrace | .rbrace | .lbracket | .rbracket |
          .comma | .semi | .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
          .ask (.scan fuel rest constants functions) (fun result => .done result)
  | .expr fuel minPrec tokens => .done (Decode.parseExprFuel fuel minPrec tokens)
  | .block fuel tokens => .done (Decode.parseBlockFuel fuel tokens)

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The scanner's equation for the recursive `scan` case, written with the public
`Decode.parseExprFuel` call that the private `parseExpr` alias denotes. -/
theorem scan_unfold (fuel : Nat) (token : Token) (rest : List Token)
    (constants functions : List String) :
    Decode.scanSelected (fuel + 1) (token :: rest) constants functions =
      (match token.kind with
       | .ident name =>
           match rest with
           | [] => Decode.scanSelected fuel [] constants functions
           | next :: tail =>
               if next.kind = .assign then
                 if constants.contains name then
                   match Decode.takeLine tail with
                   | (line, afterLine) =>
                       (do
                         let (value, leftover) ← Decode.parseExprFuel (line.length * 4 + 16) 0 line
                         if Decode.dropNewlines leftover != [] then
                           .error "trailing constant tokens"
                         else
                           let (decls, funcs) ←
                             Decode.scanSelected fuel afterLine (constants.erase name) functions
                           .ok ({ name := name, value := value,
                                  span := ⟨token.span.start,
                                    lineStop line token.span.stop⟩ } :: decls, funcs))
                 else if Decode.selectedConstantNames.contains name then
                   .error "duplicate selected constant"
                 else Decode.scanSelected fuel rest constants functions
               else if next.kind = .lparen then
                 if functions.contains name then
                   (do
                     let (decl, afterDecl) ←
                       Decode.parseFunction name token.span.start
                         ({ kind := .lparen,
                            span := ⟨token.span.stop, token.span.stop + 1⟩ } :: tail)
                     let (decls, funcs) ←
                       Decode.scanSelected fuel afterDecl constants (functions.erase name)
                     .ok (decls, decl :: funcs))
                 else Decode.scanSelected fuel rest constants functions
               else Decode.scanSelected fuel rest constants functions
       | .numeral _ => Decode.scanSelected fuel rest constants functions
       | .lparen => Decode.scanSelected fuel rest constants functions
       | .rparen => Decode.scanSelected fuel rest constants functions
       | .lbrace => Decode.scanSelected fuel rest constants functions
       | .rbrace => Decode.scanSelected fuel rest constants functions
       | .lbracket => Decode.scanSelected fuel rest constants functions
       | .rbracket => Decode.scanSelected fuel rest constants functions
       | .comma => Decode.scanSelected fuel rest constants functions
       | .semi => Decode.scanSelected fuel rest constants functions
       | .newline => Decode.scanSelected fuel rest constants functions
       | .assign => Decode.scanSelected fuel rest constants functions
       | .lt => Decode.scanSelected fuel rest constants functions
       | .le => Decode.scanSelected fuel rest constants functions
       | .gt => Decode.scanSelected fuel rest constants functions
       | .ge => Decode.scanSelected fuel rest constants functions
       | .other _ => Decode.scanSelected fuel rest constants functions) := rfl

/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .parameters 0 _ => rfl
  | .parameters (_ + 1) [] => rfl
  | .parameters (fuel + 1) (token :: rest) => by
      rw [show evaluate (.parameters (fuel + 1) (token :: rest))
            = Decode.parseParameters (fuel + 1) (token :: rest) from rfl]
      dsimp only [Slang.Derivation.Selection.body]
      rw [Decode.parseParameters]
      dsimp only []
      by_cases hr : token.kind = .rparen
      · conv => lhs; rw [if_pos hr]
        conv => rhs; rw [if_pos hr]
        rfl
      · conv => lhs; rw [if_neg hr]
        conv => rhs; rw [if_neg hr]
        cases hk : token.kind
        case ident name =>
          dsimp only []
          cases rest with
          | nil => rfl
          | cons next tail =>
              dsimp only []
              by_cases hc : next.kind = .comma
              · conv => lhs; rw [if_pos hc]
                conv => rhs; rw [if_pos hc]
                rw [Slang.Derivation.fold_ask]
                dsimp only [Slang.Derivation.Selection.evaluate,
                  Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                cases h : Decode.parseParameters fuel tail with
                | error message => rfl
                | ok pair => cases pair with | mk names tail2 => rfl
              · conv => lhs; rw [if_neg hc]
                conv => rhs; rw [if_neg hc]
                by_cases hp : next.kind = .rparen
                · conv => lhs; rw [if_pos hp]
                  conv => rhs; rw [if_pos hp]
                  rfl
                · conv => lhs; rw [if_neg hp]
                  conv => rhs; rw [if_neg hp]
                  rfl
        all_goals rfl
  | .line [] => rfl
  | .line (token :: rest) => by
      rw [show evaluate (.line (token :: rest)) = Decode.takeLine (token :: rest) from rfl]
      dsimp only [Slang.Derivation.Selection.body]
      rw [Decode.takeLine]
      dsimp only []
      by_cases hn : token.kind = .newline
      · conv => lhs; rw [if_pos hn]
        conv => rhs; rw [if_pos hn]
        rfl
      · conv => lhs; rw [if_neg hn]
        conv => rhs; rw [if_neg hn]
        rw [Slang.Derivation.fold_ask]
        dsimp only [Slang.Derivation.Selection.evaluate]
        cases h : Decode.takeLine rest with
        | mk line tail => rfl
  | .function name start tokens => by
      rw [show evaluate (.function name start tokens)
            = Decode.parseFunction name start tokens from rfl]
      dsimp only [Slang.Derivation.Selection.body]
      rw [Decode.parseFunction]
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      cases h1 : Decode.expectKind .lparen tokens with
      | error message => rfl
      | ok pair1 =>
          cases pair1 with
          | mk _ tokens1 =>
              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
              rw [Slang.Derivation.fold_ask]
              dsimp only [Slang.Derivation.Selection.evaluate,
                Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
              cases h2 : Decode.parseParameters (tokens1.length + 1) tokens1 with
              | error message => rfl
              | ok pair2 =>
                  cases pair2 with
                  | mk parameters tokens2 =>
                      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                      cases h3 : Decode.expectKind .lbrace (Decode.dropNewlines tokens2) with
                      | error message => rfl
                      | ok pair3 =>
                          cases pair3 with
                          | mk _ tokens3 =>
                              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                              rw [Slang.Derivation.fold_ask]
                              dsimp only [Slang.Derivation.Selection.evaluate,
                                Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                              cases h4 : Decode.parseBlockFuel (tokens3.length + 16) tokens3 with
                              | error message => rfl
                              | ok triple =>
                                  cases triple with
                                  | mk body rest =>
                                      cases rest with
                                      | mk close tokens4 => rfl
  | .scan 0 _ _ _ => rfl
  | .scan (_ + 1) [] _ _ => rfl
  | .scan (fuel + 1) (token :: rest) constants functions => by
      rw [show evaluate (.scan (fuel + 1) (token :: rest) constants functions)
            = Decode.scanSelected (fuel + 1) (token :: rest) constants functions from rfl]
      rw [scan_unfold]
      dsimp only [Slang.Derivation.Selection.body]
      cases hk : token.kind
      case ident name =>
        dsimp only []
        cases rest with
        | nil =>
            rw [Slang.Derivation.fold_ask]
            dsimp only [Slang.Derivation.Selection.evaluate]
            rfl
        | cons next tail =>
            dsimp only []
            by_cases has : next.kind = .assign
            · conv => lhs; rw [if_pos has]
              conv => rhs; rw [if_pos has]
              by_cases hc : constants.contains name
              · conv => lhs; rw [if_pos hc]
                conv => rhs; rw [if_pos hc]
                rw [Slang.Derivation.fold_ask]
                dsimp only [Slang.Derivation.Selection.evaluate]
                cases hline : Decode.takeLine tail with
                | mk line afterLine =>
                    dsimp only []
                    rw [Slang.Derivation.fold_ask]
                    dsimp only [Slang.Derivation.Selection.evaluate,
                      Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                    cases hexpr : Decode.parseExprFuel (line.length * 4 + 16) 0 line with
                    | error message => rfl
                    | ok pair =>
                        cases pair with
                        | mk value leftover =>
                            dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                            by_cases htrail : Decode.dropNewlines leftover != []
                            · conv => lhs; rw [if_pos htrail]
                              conv => rhs; rw [if_pos htrail]
                              rfl
                            · conv => lhs; rw [if_neg htrail]
                              conv => rhs; rw [if_neg htrail]
                              rw [Slang.Derivation.fold_ask]
                              dsimp only [Slang.Derivation.Selection.evaluate,
                                Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                              cases hscan : Decode.scanSelected fuel afterLine
                                  (constants.erase name) functions with
                              | error message => rfl
                              | ok pair2 => cases pair2 with | mk decls funcs => rfl
              · conv => lhs; rw [if_neg hc]
                conv => rhs; rw [if_neg hc]
                by_cases hdup : Decode.selectedConstantNames.contains name
                · conv => lhs; rw [if_pos hdup]
                  conv => rhs; rw [if_pos hdup]
                  rfl
                · conv => lhs; rw [if_neg hdup]
                  conv => rhs; rw [if_neg hdup]
                  rw [Slang.Derivation.fold_ask]
                  dsimp only [Slang.Derivation.Selection.evaluate]
                  rfl
            · conv => lhs; rw [if_neg has]
              conv => rhs; rw [if_neg has]
              by_cases hl : next.kind = .lparen
              · conv => lhs; rw [if_pos hl]
                conv => rhs; rw [if_pos hl]
                by_cases hfun : functions.contains name
                · conv => lhs; rw [if_pos hfun]
                  conv => rhs; rw [if_pos hfun]
                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                  rw [Slang.Derivation.fold_ask]
                  dsimp only [Slang.Derivation.Selection.evaluate,
                    Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                  cases hfn : Decode.parseFunction name token.span.start
                      ({ kind := .lparen,
                         span := ⟨token.span.stop, token.span.stop + 1⟩ } :: tail) with
                  | error message => rfl
                  | ok pair =>
                      cases pair with
                      | mk decl afterDecl =>
                          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          rw [Slang.Derivation.fold_ask]
                          dsimp only [Slang.Derivation.Selection.evaluate,
                            Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          cases hscan : Decode.scanSelected fuel afterDecl constants
                              (functions.erase name) with
                          | error message => rfl
                          | ok pair2 => cases pair2 with | mk decls funcs => rfl
                · conv => lhs; rw [if_neg hfun]
                  conv => rhs; rw [if_neg hfun]
                  rw [Slang.Derivation.fold_ask]
                  dsimp only [Slang.Derivation.Selection.evaluate]
                  rfl
              · conv => lhs; rw [if_neg hl]
                conv => rhs; rw [if_neg hl]
                rw [Slang.Derivation.fold_ask]
                dsimp only [Slang.Derivation.Selection.evaluate]
                rfl
      all_goals
        (rw [Slang.Derivation.fold_ask]
         dsimp only [Slang.Derivation.Selection.evaluate]
         rfl)
  | .expr fuel minPrec tokens => rfl
  | .block fuel tokens => rfl

/-- A checked derivation of the real scanner's result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

/-- Every parameter request has a checked derivation of the real parser's answer. -/
theorem parameters_run : ∀ (fuel : Nat) (tokens : List Token),
    Run (.parameters fuel tokens) (Decode.parseParameters fuel tokens)
  | 0, _ => Slang.Derivation.Derives.done _
  | _ + 1, [] => Slang.Derivation.Derives.done _
  | fuel + 1, token :: rest => by
      unfold Slang.Derivation.Selection.Run
      dsimp only [Slang.Derivation.Selection.body]
      rw [Decode.parseParameters.eq_3]
      dsimp only []
      by_cases hr : token.kind = .rparen
      · conv => arg 2; rw [if_pos hr]
        conv => arg 3; rw [if_pos hr]
        exact Slang.Derivation.Derives.done _
      · conv => arg 2; rw [if_neg hr]
        conv => arg 3; rw [if_neg hr]
        cases hk : token.kind
        case ident name =>
          dsimp only []
          cases rest with
          | nil => exact Slang.Derivation.Derives.done _
          | cons next tail =>
              dsimp only []
              by_cases hc : next.kind = .comma
              · conv => arg 2; rw [if_pos hc]
                conv => arg 3; rw [if_pos hc]
                refine Slang.Derivation.Derives.ask
                  (body := Slang.Derivation.Selection.body)
                  (called := parameters_run fuel tail) ?_
                cases h : Decode.parseParameters fuel tail with
                | error message => exact Slang.Derivation.Derives.done _
                | ok pair => cases pair with
                    | mk names tail2 => exact Slang.Derivation.Derives.done _
              · conv => arg 2; rw [if_neg hc]
                conv => arg 3; rw [if_neg hc]
                by_cases hp : next.kind = .rparen
                · conv => arg 2; rw [if_pos hp]
                  conv => arg 3; rw [if_pos hp]
                  exact Slang.Derivation.Derives.done _
                · conv => arg 2; rw [if_neg hp]
                  conv => arg 3; rw [if_neg hp]
                  exact Slang.Derivation.Derives.done _
        all_goals (dsimp only []; exact Slang.Derivation.Derives.done _)

/-- Every line request has a checked derivation of the real scanner's answer. -/
theorem line_run : ∀ (tokens : List Token),
    Run (.line tokens) (Decode.takeLine tokens)
  | [] => Slang.Derivation.Derives.done _
  | token :: rest => by
      unfold Slang.Derivation.Selection.Run
      dsimp only [Slang.Derivation.Selection.body]
      rw [Decode.takeLine.eq_2]
      dsimp only []
      by_cases hn : token.kind = .newline
      · conv => arg 2; rw [if_pos hn]
        conv => arg 3; rw [if_pos hn]
        exact Slang.Derivation.Derives.done _
      · conv => arg 2; rw [if_neg hn]
        conv => arg 3; rw [if_neg hn]
        refine Slang.Derivation.Derives.ask
          (body := Slang.Derivation.Selection.body)
          (called := line_run rest) ?_
        cases h : Decode.takeLine rest with
        | mk line tail => exact Slang.Derivation.Derives.done _

/-- Every function request has a checked derivation of the real parser's answer. -/
theorem function_run (name : String) (start : Nat) (tokens : List Token) :
    Run (.function name start tokens) (Decode.parseFunction name start tokens) := by
  unfold Slang.Derivation.Selection.Run
  dsimp only [Slang.Derivation.Selection.body]
  rw [Decode.parseFunction]
  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
  cases h1 : Decode.expectKind .lparen tokens with
  | error message => exact Slang.Derivation.Derives.done _
  | ok pair1 =>
      cases pair1 with
      | mk _ tokens1 =>
          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
          refine Slang.Derivation.Derives.ask
            (body := Slang.Derivation.Selection.body)
            (called := parameters_run (tokens1.length + 1) tokens1) ?_
          cases h2 : Decode.parseParameters (tokens1.length + 1) tokens1 with
          | error message => exact Slang.Derivation.Derives.done _
          | ok pair2 =>
              cases pair2 with
              | mk parameters tokens2 =>
                  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                  cases h3 : Decode.expectKind .lbrace (Decode.dropNewlines tokens2) with
                  | error message => exact Slang.Derivation.Derives.done _
                  | ok pair3 =>
                      cases pair3 with
                      | mk _ tokens3 =>
                          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          refine Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Selection.body)
                            (called := Slang.Derivation.Derives.done _) ?_
                          cases h4 : Decode.parseBlockFuel (tokens3.length + 16) tokens3 with
                          | error message => exact Slang.Derivation.Derives.done _
                          | ok triple =>
                              cases triple with
                              | mk body rest =>
                                  cases rest with
                                  | mk close tokens4 =>
                                      exact Slang.Derivation.Derives.done _

/-- Every scan request has a checked derivation of the real scanner's answer. -/
theorem scan_run : ∀ (fuel : Nat) (tokens : List Token) (constants functions : List String),
    Run (.scan fuel tokens constants functions)
      (Decode.scanSelected fuel tokens constants functions)
  | 0, _, _, _ => Slang.Derivation.Derives.done _
  | _ + 1, [], _, _ => Slang.Derivation.Derives.done _
  | fuel + 1, token :: rest, constants, functions => by
      unfold Slang.Derivation.Selection.Run
      dsimp only [Slang.Derivation.Selection.body]
      rw [scan_unfold]
      cases hk : token.kind
      case ident name =>
        dsimp only []
        cases rest with
        | nil =>
            exact Slang.Derivation.Derives.ask_done
              (body := Slang.Derivation.Selection.body)
              (scan_run fuel [] constants functions)
        | cons next tail =>
            dsimp only []
            by_cases has : next.kind = .assign
            · conv => arg 2; rw [if_pos has]
              conv => arg 3; rw [if_pos has]
              by_cases hc : constants.contains name
              · conv => arg 2; rw [if_pos hc]
                conv => arg 3; rw [if_pos hc]
                refine Slang.Derivation.Derives.ask
                  (body := Slang.Derivation.Selection.body)
                  (called := line_run tail) ?_
                cases hline : Decode.takeLine tail with
                | mk line afterLine =>
                    dsimp only []
                    refine Slang.Derivation.Derives.ask
                      (body := Slang.Derivation.Selection.body)
                      (called := Slang.Derivation.Derives.done _) ?_
                    cases hexpr : Decode.parseExprFuel (line.length * 4 + 16) 0 line with
                    | error message => exact Slang.Derivation.Derives.done _
                    | ok pair =>
                        cases pair with
                        | mk value leftover =>
                            dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                            by_cases htrail : Decode.dropNewlines leftover != []
                            · conv => arg 2; rw [if_pos htrail]
                              conv => arg 3; rw [if_pos htrail]
                              exact Slang.Derivation.Derives.done _
                            · conv => arg 2; rw [if_neg htrail]
                              conv => arg 3; rw [if_neg htrail]
                              refine Slang.Derivation.Derives.ask
                                (body := Slang.Derivation.Selection.body)
                                (called := scan_run fuel afterLine
                                  (constants.erase name) functions) ?_
                              cases hscan : Decode.scanSelected fuel afterLine
                                  (constants.erase name) functions with
                              | error message => exact Slang.Derivation.Derives.done _
                              | ok pair2 => cases pair2 with
                                  | mk decls funcs =>
                                      exact Slang.Derivation.Derives.done _
              · conv => arg 2; rw [if_neg hc]
                conv => arg 3; rw [if_neg hc]
                by_cases hdup : Decode.selectedConstantNames.contains name
                · conv => arg 2; rw [if_pos hdup]
                  conv => arg 3; rw [if_pos hdup]
                  exact Slang.Derivation.Derives.done _
                · conv => arg 2; rw [if_neg hdup]
                  conv => arg 3; rw [if_neg hdup]
                  exact Slang.Derivation.Derives.ask_done
                    (body := Slang.Derivation.Selection.body)
                    (scan_run fuel (next :: tail) constants functions)
            · conv => arg 2; rw [if_neg has]
              conv => arg 3; rw [if_neg has]
              by_cases hl : next.kind = .lparen
              · conv => arg 2; rw [if_pos hl]
                conv => arg 3; rw [if_pos hl]
                by_cases hfun : functions.contains name
                · conv => arg 2; rw [if_pos hfun]
                  conv => arg 3; rw [if_pos hfun]
                  refine Slang.Derivation.Derives.ask
                    (body := Slang.Derivation.Selection.body)
                    (called := function_run name token.span.start
                      ({ kind := .lparen,
                         span := ⟨token.span.stop, token.span.stop + 1⟩ } :: tail)) ?_
                  cases hfn : Decode.parseFunction name token.span.start
                      ({ kind := .lparen,
                         span := ⟨token.span.stop, token.span.stop + 1⟩ } :: tail) with
                  | error message => exact Slang.Derivation.Derives.done _
                  | ok pair =>
                      cases pair with
                      | mk decl afterDecl =>
                          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
                          refine Slang.Derivation.Derives.ask
                            (body := Slang.Derivation.Selection.body)
                            (called := scan_run fuel afterDecl constants
                              (functions.erase name)) ?_
                          cases hscan : Decode.scanSelected fuel afterDecl constants
                              (functions.erase name) with
                          | error message => exact Slang.Derivation.Derives.done _
                          | ok pair2 => cases pair2 with
                              | mk decls funcs =>
                                  exact Slang.Derivation.Derives.done _
                · conv => arg 2; rw [if_neg hfun]
                  conv => arg 3; rw [if_neg hfun]
                  exact Slang.Derivation.Derives.ask_done
                    (body := Slang.Derivation.Selection.body)
                    (scan_run fuel (next :: tail) constants functions)
              · conv => arg 2; rw [if_neg hl]
                conv => arg 3; rw [if_neg hl]
                exact Slang.Derivation.Derives.ask_done
                  (body := Slang.Derivation.Selection.body)
                  (scan_run fuel (next :: tail) constants functions)
      all_goals
        (dsimp only []
         exact Slang.Derivation.Derives.ask_done
           (body := Slang.Derivation.Selection.body)
           (scan_run fuel rest constants functions))

/-- Transfer a `Grammar` expression derivation to the `Selection` expression leaf.

`Selection.body (.expr fuel minPrec tokens)` is the leaf
`.done (Decode.parseExprFuel fuel minPrec tokens)`, the real call that
`Grammar.evaluate (.expr fuel minPrec tokens)` names. The child's checked
equality is transported into the parent `.done` by `Derives.done_eq`. -/
theorem expr_of_grammar {fuel minPrec : Nat} {tokens : List Token}
    {answer : Result (Expr × List Token)}
    (derivation : Grammar.Run (.expr fuel minPrec tokens) answer) :
    Run (.expr fuel minPrec tokens) answer :=
  Slang.Derivation.Derives.done_eq (Grammar.Run.sound derivation)

/-- Transfer a `Statements` block derivation to the `Selection` block leaf.

`Selection.body (.block fuel tokens)` is the leaf
`.done (Decode.parseBlockFuel fuel tokens)`, the real call that
`Statements.evaluate (.block fuel tokens)` names. -/
theorem block_of_statements {fuel : Nat} {tokens : List Token}
    {answer : Result (List Stmt × Token × List Token)}
    (derivation : Statements.Run (.block fuel tokens) answer) :
    Run (.block fuel tokens) answer :=
  Slang.Derivation.Derives.done_eq (Statements.Run.sound derivation)

/-- The concrete `Grammar` expression leaf at fuel zero, transferred into `Selection`. -/
theorem expr_zero_run :
    Run (.expr 0 0 ([] : List Token))
      (Except.error "expression nesting limit reached") :=
  expr_of_grammar (Grammar.expr_run 0 0 [])

/-- The concrete `Statements` block leaf at fuel zero, transferred into `Selection`. -/
theorem block_zero_run :
    Run (.block 0 ([] : List Token))
      (Except.error "statement nesting limit reached") :=
  block_of_statements (Statements.block_run 0 [])

/-- A wrong answer for the expression leaf is not derivable through the adapter. -/
theorem expr_zero_wrong :
    ¬ Run (.expr 0 0 ([] : List Token)) (Except.ok (Expr.literal 0, [])) := by
  intro derivation
  have sound := Run.sound derivation
  have limit : evaluate (.expr 0 0 ([] : List Token)) =
      Except.error "expression nesting limit reached" := rfl
  rw [limit] at sound
  cases sound

/-- A wrong answer for the block leaf is not derivable through the adapter. -/
theorem block_zero_wrong :
    ¬ Run (.block 0 ([] : List Token))
      (Except.ok ([], ({ kind := .rbrace, span := ⟨0, 0⟩ } : Token), [])) := by
  intro derivation
  have sound := Run.sound derivation
  have limit : evaluate (.block 0 ([] : List Token)) =
      Except.error "statement nesting limit reached" := rfl
  rw [limit] at sound
  cases sound

end Slang.Derivation.Selection
