import Slang.Audit.Empty
import Slang.Derivation.Steps
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 4096

namespace Slang.Derivation.Grammar

open Slang Slang.Decode

/-- Requests name the recursive work of the expression parser instead of evaluating it. -/
inductive Request where
  | expr (fuel minPrec : Nat) (tokens : List Token)
  | primary (fuel : Nat) (tokens : List Token)
  | args (fuel : Nat) (tokens : List Token)
  | binary (fuel minPrec : Nat) (left : Expr) (tokens : List Token)

/-- Answer type per request. -/
abbrev Answer : Request → Type
  | .expr _ _ _ => Result (Expr × List Token)
  | .primary _ _ => Result (Expr × List Token)
  | .args _ _ => Result (List Expr × List Token)
  | .binary _ _ _ _ => Result (Expr × List Token)

/-- The real expression parser, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .expr fuel minPrec tokens => Decode.parseExprFuel fuel minPrec tokens
  | .primary fuel tokens => Decode.parsePrimaryFuel fuel tokens
  | .args fuel tokens => Decode.parseArgsFuel fuel tokens
  | .binary fuel minPrec left tokens => Decode.parseBinaryFuel fuel minPrec left tokens

/-- One unfolding of `parseExprFuel`, exposing recursive calls as requests. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .expr 0 _ _ => .done (.error "expression nesting limit reached")
  | .expr (fuel + 1) minPrec tokens =>
      .ask (.primary fuel tokens) (fun primary =>
        match primary with
        | .ok (left, rest) => .ask (.binary fuel minPrec left rest) (fun result => .done result)
        | .error message => .done (.error message))
  | .primary 0 _ => .done (.error "expression nesting limit reached")
  | .primary (fuel + 1) tokens =>
      match Decode.dropNewlines tokens with
      | [] => .done (.error "expected expression at end of input")
      | token :: rest =>
          match token.kind with
          | .numeral value => .done (.ok (.literal value, rest))
          | .ident name =>
              if name = "memory" then
                match Decode.expectKind .lbracket rest with
                | .error message => .done (.error message)
                | .ok (_, rest1) =>
                    .ask (.expr fuel 0 rest1) (fun r =>
                      match r with
                      | .error message => .done (.error message)
                      | .ok (address, rest2) =>
                          match Decode.expectKind .rbracket rest2 with
                          | .error message => .done (.error message)
                          | .ok (_, rest3) => .done (.ok (.memory address, rest3)))
              else
                match rest with
                | next :: tail =>
                    if next.kind = .lparen then
                      .ask (.args fuel tail) (fun r =>
                        match r with
                        | .error message => .done (.error message)
                        | .ok (args, rest1) => .done (.ok (.call name args, rest1)))
                    else .done (.ok (.name name, rest))
                | [] => .done (.ok (.name name, []))
          | .lparen =>
              .ask (.expr fuel 0 rest) (fun r =>
                match r with
                | .error message => .done (.error message)
                | .ok (value, rest1) =>
                    match Decode.expectKind .rparen rest1 with
                    | .error message => .done (.error message)
                    | .ok (_, rest2) => .done (.ok (value, rest2)))
          | .rparen | .lbrace | .rbrace | .lbracket | .rbracket | .comma | .semi |
              .newline | .assign | .lt | .le | .gt | .ge | .other _ =>
              .done (.error "unsupported expression token")
  | .args 0 _ => .done (.error "call argument nesting limit reached")
  | .args (fuel + 1) tokens =>
      match Decode.dropNewlines tokens with
      | [] => .done (.error "unterminated call arguments")
      | token :: rest =>
          if token.kind = .rparen then .done (.ok ([], rest)) else
            .ask (.expr fuel 0 (token :: rest)) (fun r =>
              match r with
              | .error message => .done (.error message)
              | .ok (arg, rest1) =>
                  match Decode.dropNewlines rest1 with
                  | next :: tail =>
                      if next.kind = .comma then
                        .ask (.args fuel tail) (fun r2 =>
                          match r2 with
                          | .error message => .done (.error message)
                          | .ok (args, tail2) => .done (.ok (arg :: args, tail2)))
                      else if next.kind = .rparen then .done (.ok ([arg], tail))
                      else .done (.error "expected comma or closing parenthesis")
                  | [] => .done (.error "unterminated call arguments"))
  | .binary 0 _ _ _ => .done (.error "expression operator limit reached")
  | .binary (fuel + 1) minPrec left tokens =>
      match Decode.dropNewlines tokens with
      | [] => .done (.ok (left, []))
      | token :: rest =>
          let ordinary : Layer Request Answer (Result (Expr × List Token)) :=
            match Decode.binOp token.kind with
            | some (op, prec) =>
                if prec < minPrec then .done (.ok (left, tokens)) else
                  .ask (.expr fuel (prec + 1) rest) (fun r =>
                    match r with
                    | .error message => .done (.error message)
                    | .ok (right, rest2) =>
                        .ask (.binary fuel minPrec (.binary op left right) rest2) (fun r2 => .done r2))
            | none =>
                if token.kind = .ident "if" then
                  if 0 < minPrec then .done (.ok (left, tokens)) else
                    .ask (.expr fuel 0 rest) (fun r =>
                      match r with
                      | .error message => .done (.error message)
                      | .ok (condition, rest2) =>
                          match Decode.dropNewlines rest2 with
                          | next :: tail =>
                              if next.kind = .ident "else" then
                                .ask (.expr fuel 0 tail) (fun r3 =>
                                  match r3 with
                                  | .error message => .done (.error message)
                                  | .ok (elseValue, tail2) =>
                                      .done (.ok (.conditional left condition elseValue, tail2)))
                              else .done (.error "expected inline else")
                          | [] => .done (.error "unterminated inline conditional"))
                else .done (.ok (left, tokens))
          if token.kind = .ident "is" then
            match rest with
            | next :: tail =>
                if next.kind = .ident "not" then
                  if 3 < minPrec then .done (.ok (left, tokens)) else
                    .ask (.expr fuel 4 tail) (fun r =>
                      match r with
                      | .error message => .done (.error message)
                      | .ok (right, rest2) =>
                          .ask (.binary fuel minPrec (.binary .ne left right) rest2)
                            (fun r2 => .done r2))
                else ordinary
            | [] => ordinary
          else ordinary

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The operator tail of `parseBinaryFuel` commutes with the layer fold. -/
theorem binary_ordinary_eq (n minPrec : Nat) (left : Expr) (tokens rest : List Token)
    (token : Token) :
    (match Decode.binOp token.kind with
      | some (op, prec) =>
          if prec < minPrec then .ok (left, tokens) else do
            let (right, rest2) ← Decode.parseExprFuel n (prec + 1) rest
            Decode.parseBinaryFuel n minPrec (.binary op left right) rest2
      | none =>
          if token.kind = .ident "if" then
            if 0 < minPrec then .ok (left, tokens) else do
              let (condition, rest2) ← Decode.parseExprFuel n 0 rest
              match Decode.dropNewlines rest2 with
              | next :: tail =>
                  if next.kind = .ident "else" then do
                    let (elseValue, tail2) ← Decode.parseExprFuel n 0 tail
                    .ok (.conditional left condition elseValue, tail2)
                  else .error "expected inline else"
              | [] => .error "unterminated inline conditional"
          else .ok (left, tokens))
    = fold evaluate
        (match Decode.binOp token.kind with
          | some (op, prec) =>
              if prec < minPrec then .done (.ok (left, tokens)) else
                .ask (.expr n (prec + 1) rest) (fun r =>
                  match r with
                  | .error message => .done (.error message)
                  | .ok (right, rest2) =>
                      .ask (.binary n minPrec (.binary op left right) rest2)
                        (fun r2 => .done r2))
          | none =>
              if token.kind = .ident "if" then
                if 0 < minPrec then .done (.ok (left, tokens)) else
                  .ask (.expr n 0 rest) (fun r =>
                    match r with
                    | .error message => .done (.error message)
                    | .ok (condition, rest2) =>
                        match Decode.dropNewlines rest2 with
                        | next :: tail =>
                            if next.kind = .ident "else" then
                              .ask (.expr n 0 tail) (fun r3 =>
                                match r3 with
                                | .error message => .done (.error message)
                                | .ok (elseValue, tail2) =>
                                    .done (.ok (.conditional left condition elseValue, tail2)))
                            else .done (.error "expected inline else")
                        | [] => .done (.error "unterminated inline conditional"))
                else .done (.ok (left, tokens))) := by
  cases hb : Decode.binOp token.kind with
  | none =>
      dsimp only []
      by_cases hif : token.kind = .ident "if"
      · repeat rw [if_pos hif]
        by_cases hprec0 : 0 < minPrec
        · repeat rw [if_pos hprec0]
          rfl
        · repeat rw [if_neg hprec0]
          dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
            Bind.bind, Except.bind]
          cases he : Decode.parseExprFuel n 0 rest with
          | error message => rfl
          | ok pair =>
              cases pair with
              | mk condition rest2 =>
                  dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                    Bind.bind, Except.bind]
                  cases hd2 : Decode.dropNewlines rest2 with
                  | nil => rfl
                  | cons next tail =>
                      dsimp only []
                      by_cases hnot2 : next.kind = .ident "else"
                      · repeat rw [if_pos hnot2]
                        dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                          Bind.bind, Except.bind]
                        cases he2 : Decode.parseExprFuel n 0 tail with
                        | error message => rfl
                        | ok pair2 => cases pair2 with | mk elseValue tail2 => rfl
                      · repeat rw [if_neg hnot2]
                        rfl
      · repeat rw [if_neg hif]
        rfl
  | some pair =>
      cases pair with
      | mk op prec =>
          dsimp only []
          by_cases hprec : prec < minPrec
          · repeat rw [if_pos hprec]
            rfl
          · repeat rw [if_neg hprec]
            dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
              Bind.bind, Except.bind]
            cases he : Decode.parseExprFuel n (prec + 1) rest with
            | error message => rfl
            | ok pair2 =>
                cases pair2 with
                | mk right rest2 =>
                    dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                      Bind.bind, Except.bind]


/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .expr 0 _ _ => rfl
  | .expr (n + 1) minPrec tokens => by
      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.Grammar.body,
        Slang.Derivation.fold, Slang.Decode.parseExprFuel]
      cases h : Decode.parsePrimaryFuel n tokens with
      | error message => rfl
      | ok pair => cases pair with | mk left rest => rfl
  | .primary 0 _ => rfl
  | .primary (n + 1) tokens => by
      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.Grammar.body,
        Slang.Decode.parsePrimaryFuel, Slang.Derivation.fold, Bind.bind, Except.bind]
      cases hd : Decode.dropNewlines tokens with
      | nil => rfl
      | cons token rest =>
          dsimp only []
          cases hk : token.kind
          case numeral value => rfl
          case ident name =>
              dsimp only []
              by_cases hm : name = "memory"
              · repeat rw [if_pos hm]
                cases he : Decode.expectKind .lbracket rest with
                | error message => rfl
                | ok pair =>
                    cases pair with
                    | mk _ rest1 =>
                        dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                          Bind.bind, Except.bind]
                        cases hx : Decode.parseExprFuel n 0 rest1 with
                        | error message => rfl
                        | ok pair2 =>
                            cases pair2 with
                            | mk address rest2 =>
                                dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                                  Bind.bind, Except.bind]
                                cases he2 : Decode.expectKind .rbracket rest2 with
                                | error message => rfl
                                | ok pair3 => cases pair3 with | mk _ rest3 => rfl
              · repeat rw [if_neg hm]
                cases hr : rest with
                | nil => rfl
                | cons next tail =>
                    dsimp only []
                    by_cases hl : next.kind = .lparen
                    · repeat rw [if_pos hl]
                      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                        Bind.bind, Except.bind]
                      cases ha : Decode.parseArgsFuel n tail with
                      | error message => rfl
                      | ok pair => cases pair with | mk args rest1 => rfl
                    · repeat rw [if_neg hl]
                      rfl
          case lparen =>
              dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                Bind.bind, Except.bind]
              cases hx : Decode.parseExprFuel n 0 rest with
              | error message => rfl
              | ok pair =>
                  cases pair with
                  | mk value rest1 =>
                      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                        Bind.bind, Except.bind]
                      cases he : Decode.expectKind .rparen rest1 with
                      | error message => rfl
                      | ok pair2 => cases pair2 with | mk _ rest2 => rfl
          all_goals rfl
  | .args 0 _ => rfl
  | .args (n + 1) tokens => by
      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.Grammar.body,
        Slang.Decode.parseArgsFuel, Slang.Derivation.fold, Bind.bind, Except.bind]
      cases hd : Decode.dropNewlines tokens with
      | nil => rfl
      | cons token rest =>
          dsimp only []
          by_cases hr : token.kind = .rparen
          · repeat rw [if_pos hr]
            rfl
          · repeat rw [if_neg hr]
            dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
              Bind.bind, Except.bind]
            cases he : Decode.parseExprFuel n 0 (token :: rest) with
            | error message => rfl
            | ok pair =>
                cases pair with
                | mk arg rest1 =>
                    dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                      Bind.bind, Except.bind]
                    cases hd2 : Decode.dropNewlines rest1 with
                    | nil => rfl
                    | cons next tail =>
                        dsimp only []
                        by_cases hc : next.kind = .comma
                        · repeat rw [if_pos hc]
                          dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                            Bind.bind, Except.bind]
                          cases ha : Decode.parseArgsFuel n tail with
                          | error message => rfl
                          | ok pair2 => cases pair2 with | mk args tail2 => rfl
                        · repeat rw [if_neg hc]
                          by_cases hp : next.kind = .rparen
                          · repeat rw [if_pos hp]
                            rfl
                          · repeat rw [if_neg hp]
                            rfl
  | .binary 0 _ _ _ => rfl
  | .binary (n + 1) minPrec left tokens => by
      dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.Grammar.body,
        Slang.Decode.parseBinaryFuel, Slang.Derivation.fold, Bind.bind, Except.bind]
      cases hd : Decode.dropNewlines tokens with
      | nil => rfl
      | cons token rest =>
          dsimp only []
          by_cases his : token.kind = .ident "is"
          · repeat rw [if_pos his]
            cases hr : rest with
            | nil => dsimp only []; exact binary_ordinary_eq n minPrec left tokens _ token
            | cons next tail =>
                dsimp only []
                by_cases hnot : next.kind = .ident "not"
                · repeat rw [if_pos hnot]
                  by_cases hprec3 : 3 < minPrec
                  · repeat rw [if_pos hprec3]
                    rfl
                  · repeat rw [if_neg hprec3]
                    dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                      Bind.bind, Except.bind]
                    cases he3 : Decode.parseExprFuel n 4 tail with
                    | error message => rfl
                    | ok pair4 =>
                        cases pair4 with
                        | mk right rest2 =>
                            dsimp only [Slang.Derivation.Grammar.evaluate, Slang.Derivation.fold,
                              Bind.bind, Except.bind]
                · repeat rw [if_neg hnot]
                  exact binary_ordinary_eq n minPrec left tokens _ token
          · repeat rw [if_neg his]
            exact binary_ordinary_eq n minPrec left tokens _ token


/-- A checked derivation of the real parser's result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

mutual
  /-- Every expression request has a checked derivation of the real parser's answer. -/
  theorem expr_run : ∀ (fuel minPrec : Nat) (tokens : List Token),
      Run (.expr fuel minPrec tokens) (Decode.parseExprFuel fuel minPrec tokens)
    | 0, _, _ => Derives.done _
    | fuel + 1, minPrec, tokens => by
        unfold Slang.Derivation.Grammar.Run Slang.Derivation.Grammar.body
        dsimp only [Slang.Decode.parseExprFuel, Bind.bind, Except.bind]
        refine Derives.ask (body := Slang.Derivation.Grammar.body)
          (called := primary_run fuel tokens) ?_
        cases h : Decode.parsePrimaryFuel fuel tokens with
        | error message =>
            dsimp only [Bind.bind, Except.bind]
            exact Derives.done _
        | ok pair =>
            cases pair with
            | mk left rest =>
                dsimp only [Bind.bind, Except.bind]
                exact Derives.ask_done (body := Slang.Derivation.Grammar.body)
                  (binary_run fuel minPrec left rest)

  /-- Every primary request has a checked derivation of the real parser's answer. -/
  theorem primary_run : ∀ (fuel : Nat) (tokens : List Token),
      Run (.primary fuel tokens) (Decode.parsePrimaryFuel fuel tokens)
    | 0, _ => Derives.done _
    | fuel + 1, tokens => by
        unfold Slang.Derivation.Grammar.Run Slang.Derivation.Grammar.body
        dsimp only [Slang.Decode.parsePrimaryFuel, Bind.bind, Except.bind]
        cases hd : Decode.dropNewlines tokens with
        | nil => dsimp only []; exact Derives.done _
        | cons token rest =>
            dsimp only []
            cases hk : token.kind
            case numeral value => dsimp only []; exact Derives.done _
            case ident name =>
                dsimp only []
                by_cases hm : name = "memory"
                · repeat rw [if_pos hm]
                  cases he : Decode.expectKind .lbracket rest with
                  | error message => dsimp only []; exact Derives.done _
                  | ok pair =>
                      cases pair with
                      | mk _ rest1 =>
                          dsimp only []
                          refine Derives.ask (body := Slang.Derivation.Grammar.body)
                            (called := expr_run fuel 0 rest1) ?_
                          cases hx : Decode.parseExprFuel fuel 0 rest1 with
                          | error message =>
                              dsimp only [Bind.bind, Except.bind]
                              exact Derives.done _
                          | ok pair2 =>
                              cases pair2 with
                              | mk address rest2 =>
                                  dsimp only [Bind.bind, Except.bind]
                                  cases he2 : Decode.expectKind .rbracket rest2 with
                                  | error message =>
                                      dsimp only [Bind.bind, Except.bind]
                                      exact Derives.done _
                                  | ok pair3 =>
                                      cases pair3 with
                                      | mk _ rest3 =>
                                          dsimp only [Bind.bind, Except.bind]
                                          exact Derives.done _
                · repeat rw [if_neg hm]
                  cases hr : rest with
                  | nil => dsimp only []; exact Derives.done _
                  | cons next tail =>
                      dsimp only []
                      by_cases hl : next.kind = .lparen
                      · repeat rw [if_pos hl]
                        refine Derives.ask (body := Slang.Derivation.Grammar.body)
                          (called := args_run fuel tail) ?_
                        cases ha : Decode.parseArgsFuel fuel tail with
                        | error message =>
                            dsimp only [Bind.bind, Except.bind]
                            exact Derives.done _
                        | ok pair =>
                            cases pair with
                            | mk args rest1 =>
                                dsimp only [Bind.bind, Except.bind]
                                exact Derives.done _
                      · repeat rw [if_neg hl]
                        exact Derives.done _
            case lparen =>
                dsimp only []
                refine Derives.ask (body := Slang.Derivation.Grammar.body)
                  (called := expr_run fuel 0 rest) ?_
                cases hx : Decode.parseExprFuel fuel 0 rest with
                | error message =>
                    dsimp only [Bind.bind, Except.bind]
                    exact Derives.done _
                | ok pair =>
                    cases pair with
                    | mk value rest1 =>
                        dsimp only [Bind.bind, Except.bind]
                        cases he : Decode.expectKind .rparen rest1 with
                        | error message =>
                            dsimp only [Bind.bind, Except.bind]
                            exact Derives.done _
                        | ok pair2 =>
                            cases pair2 with
                            | mk _ rest2 =>
                                dsimp only [Bind.bind, Except.bind]
                                exact Derives.done _
            all_goals (dsimp only []; exact Derives.done _)

  /-- Every argument request has a checked derivation of the real parser's answer. -/
  theorem args_run : ∀ (fuel : Nat) (tokens : List Token),
      Run (.args fuel tokens) (Decode.parseArgsFuel fuel tokens)
    | 0, _ => Derives.done _
    | fuel + 1, tokens => by
        unfold Slang.Derivation.Grammar.Run Slang.Derivation.Grammar.body
        dsimp only [Slang.Decode.parseArgsFuel, Bind.bind, Except.bind]
        cases hd : Decode.dropNewlines tokens with
        | nil => dsimp only []; exact Derives.done _
        | cons token rest =>
            dsimp only []
            by_cases hr : token.kind = .rparen
            · repeat rw [if_pos hr]
              exact Derives.done _
            · repeat rw [if_neg hr]
              refine Derives.ask (body := Slang.Derivation.Grammar.body)
                (called := expr_run fuel 0 (token :: rest)) ?_
              cases he : Decode.parseExprFuel fuel 0 (token :: rest) with
              | error message =>
                  dsimp only [Bind.bind, Except.bind]
                  exact Derives.done _
              | ok pair =>
                  cases pair with
                  | mk arg rest1 =>
                      dsimp only [Bind.bind, Except.bind]
                      cases hd2 : Decode.dropNewlines rest1 with
                      | nil =>
                          dsimp only [Bind.bind, Except.bind]
                          exact Derives.done _
                      | cons next tail =>
                          dsimp only []
                          by_cases hc : next.kind = .comma
                          · repeat rw [if_pos hc]
                            refine Derives.ask (body := Slang.Derivation.Grammar.body)
                              (called := args_run fuel tail) ?_
                            cases ha : Decode.parseArgsFuel fuel tail with
                            | error message =>
                                dsimp only [Bind.bind, Except.bind]
                                exact Derives.done _
                            | ok pair2 =>
                                cases pair2 with
                                | mk args tail2 =>
                                    dsimp only [Bind.bind, Except.bind]
                                    exact Derives.done _
                          · repeat rw [if_neg hc]
                            by_cases hp : next.kind = .rparen
                            · repeat rw [if_pos hp]
                              exact Derives.done _
                            · repeat rw [if_neg hp]
                              exact Derives.done _

  /-- Every operator request has a checked derivation of the real parser's answer. -/
  theorem binary_run : ∀ (fuel minPrec : Nat) (left : Expr) (tokens : List Token),
      Run (.binary fuel minPrec left tokens) (Decode.parseBinaryFuel fuel minPrec left tokens)
    | 0, _, _, _ => Derives.done _
    | fuel + 1, minPrec, left, tokens => by
        unfold Slang.Derivation.Grammar.Run Slang.Derivation.Grammar.body
        dsimp only [Slang.Decode.parseBinaryFuel, Bind.bind, Except.bind]
        cases hd : Decode.dropNewlines tokens with
        | nil => dsimp only []; exact Derives.done _
        | cons token rest =>
            dsimp only []
            have ordinary_run : ∀ (rest : List Token),
                Derives body
                  (match Decode.binOp token.kind with
                    | some (op, prec) =>
                        if prec < minPrec then .done (.ok (left, tokens)) else
                          .ask (.expr fuel (prec + 1) rest) (fun r =>
                            match r with
                            | .error message => .done (.error message)
                            | .ok (right, rest2) =>
                                .ask (.binary fuel minPrec (.binary op left right) rest2)
                                  (fun r2 => .done r2))
                    | none =>
                        if token.kind = .ident "if" then
                          if 0 < minPrec then .done (.ok (left, tokens)) else
                            .ask (.expr fuel 0 rest) (fun r =>
                              match r with
                              | .error message => .done (.error message)
                              | .ok (condition, rest2) =>
                                  match Decode.dropNewlines rest2 with
                                  | next :: tail =>
                                      if next.kind = .ident "else" then
                                        .ask (.expr fuel 0 tail) (fun r3 =>
                                          match r3 with
                                          | .error message => .done (.error message)
                                          | .ok (elseValue, tail2) =>
                                              .done (.ok (.conditional left condition elseValue, tail2)))
                                      else .done (.error "expected inline else")
                                  | [] => .done (.error "unterminated inline conditional"))
                          else .done (.ok (left, tokens)))
                  (match Decode.binOp token.kind with
                    | some (op, prec) =>
                        if prec < minPrec then .ok (left, tokens) else do
                          let (right, rest2) ← Decode.parseExprFuel fuel (prec + 1) rest
                          Decode.parseBinaryFuel fuel minPrec (.binary op left right) rest2
                    | none =>
                        if token.kind = .ident "if" then
                          if 0 < minPrec then .ok (left, tokens) else do
                            let (condition, rest2) ← Decode.parseExprFuel fuel 0 rest
                            match Decode.dropNewlines rest2 with
                            | next :: tail =>
                                if next.kind = .ident "else" then do
                                  let (elseValue, tail2) ← Decode.parseExprFuel fuel 0 tail
                                  .ok (.conditional left condition elseValue, tail2)
                                else .error "expected inline else"
                            | [] => .error "unterminated inline conditional"
                        else .ok (left, tokens)) := by
              intro rest
              cases hb : Decode.binOp token.kind with
              | none =>
                  dsimp only []
                  by_cases hif : token.kind = .ident "if"
                  · repeat rw [if_pos hif]
                    by_cases hprec0 : 0 < minPrec
                    · repeat rw [if_pos hprec0]
                      exact Derives.done _
                    · repeat rw [if_neg hprec0]
                      refine Derives.ask (body := Slang.Derivation.Grammar.body)
                        (called := expr_run fuel 0 rest) ?_
                      cases he : Decode.parseExprFuel fuel 0 rest with
                      | error message =>
                          dsimp only [Bind.bind, Except.bind]
                          exact Derives.done _
                      | ok pair =>
                          cases pair with
                          | mk condition rest2 =>
                              dsimp only [Bind.bind, Except.bind]
                              cases hd2 : Decode.dropNewlines rest2 with
                              | nil =>
                                  dsimp only [Bind.bind, Except.bind]
                                  exact Derives.done _
                              | cons next tail =>
                                  dsimp only []
                                  by_cases hnot2 : next.kind = .ident "else"
                                  · repeat rw [if_pos hnot2]
                                    refine Derives.ask (body := Slang.Derivation.Grammar.body)
                                      (called := expr_run fuel 0 tail) ?_
                                    cases he2 : Decode.parseExprFuel fuel 0 tail with
                                    | error message =>
                                        dsimp only [Bind.bind, Except.bind]
                                        exact Derives.done _
                                    | ok pair2 =>
                                        cases pair2 with
                                        | mk elseValue tail2 =>
                                            dsimp only [Bind.bind, Except.bind]
                                            exact Derives.done _
                                  · repeat rw [if_neg hnot2]
                                    exact Derives.done _
                  · repeat rw [if_neg hif]
                    exact Derives.done _
              | some pair =>
                  cases pair with
                  | mk op prec =>
                      dsimp only []
                      by_cases hprec : prec < minPrec
                      · repeat rw [if_pos hprec]
                        exact Derives.done _
                      · repeat rw [if_neg hprec]
                        refine Derives.ask (body := Slang.Derivation.Grammar.body)
                          (called := expr_run fuel (prec + 1) rest) ?_
                        cases he : Decode.parseExprFuel fuel (prec + 1) rest with
                        | error message =>
                            dsimp only [Bind.bind, Except.bind]
                            exact Derives.done _
                        | ok pair2 =>
                            cases pair2 with
                            | mk right rest2 =>
                                dsimp only [Bind.bind, Except.bind]
                                refine Derives.ask (body := Slang.Derivation.Grammar.body)
                                  (called := binary_run fuel minPrec (.binary op left right) rest2) ?_
                                cases hb2 : Decode.parseBinaryFuel fuel minPrec
                                    (.binary op left right) rest2 with
                                | error message => exact Derives.done _
                                | ok pair3 =>
                                    cases pair3 with
                                    | mk v r => exact Derives.done _
            by_cases his : token.kind = .ident "is"
            · repeat rw [if_pos his]
              cases hr : rest with
              | nil => dsimp only []; exact ordinary_run []
              | cons next tail =>
                  dsimp only []
                  by_cases hnot : next.kind = .ident "not"
                  · repeat rw [if_pos hnot]
                    by_cases hprec3 : 3 < minPrec
                    · repeat rw [if_pos hprec3]
                      exact Derives.done _
                    · repeat rw [if_neg hprec3]
                      refine Derives.ask (body := Slang.Derivation.Grammar.body)
                        (called := expr_run fuel 4 tail) ?_
                      cases he3 : Decode.parseExprFuel fuel 4 tail with
                      | error message =>
                          dsimp only [Bind.bind, Except.bind]
                          exact Derives.done _
                      | ok pair4 =>
                          cases pair4 with
                          | mk right rest2 =>
                              dsimp only [Bind.bind, Except.bind]
                              refine Derives.ask (body := Slang.Derivation.Grammar.body)
                                (called := binary_run fuel minPrec (.binary .ne left right) rest2) ?_
                              cases hb3 : Decode.parseBinaryFuel fuel minPrec
                                  (.binary .ne left right) rest2 with
                              | error message => exact Derives.done _
                              | ok pair5 =>
                                  cases pair5 with
                                  | mk v r => exact Derives.done _
                  · repeat rw [if_neg hnot]
                    exact ordinary_run (next :: tail)
            · repeat rw [if_neg his]
              exact ordinary_run rest
end


end Slang.Derivation.Grammar