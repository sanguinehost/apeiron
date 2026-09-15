import Slang.Audit.Empty
import Slang.Derivation.Steps
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000

namespace Slang.Derivation.Lexing

open Slang Slang.Decode

/-- Requests name the recursive work of the tokenizer instead of evaluating it. -/
inductive Request where
  | chars (bytes : List UInt8)
  | lex (fuel : Nat) (chars : List Char) (pos : Nat) (tokens : List Token)
  | tokenize (bytes : ByteArray)

/-- Answer type per request. -/
abbrev Answer : Request → Type
  | .chars _ => Result (List Char)
  | .lex _ _ _ _ => Result (List Token × Nat)
  | .tokenize _ => Result (List Token × Nat)

/-- The real implementation, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .chars bytes => Decode.decodeUtf8Chars bytes
  | .lex fuel chars pos tokens => Slang.Decode.lexChars fuel chars pos tokens
  | .tokenize bytes => Decode.tokenize bytes

/-- A recursive scan appears as a request rather than as an evaluated call. -/
def lexAsk (fuel : Nat) (chars : List Char) (pos : Nat) (tokens : List Token) :
    Layer Request Answer (Result (List Token × Nat)) :=
  .ask (.lex fuel chars pos tokens) (fun result => .done result)

/-- One unfolding of the implementation, exposing recursive calls as requests. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .chars bytes => .done (Decode.decodeUtf8Chars bytes)
  | .tokenize bytes =>
      .ask (.chars bytes.data.toList) (fun result =>
        match result with
        | .ok chars => lexAsk (chars.length + 1) chars 0 []
        | .error message => .done (.error message))
  | .lex fuel chars pos tokens =>
      match fuel, chars with
      | 0, _ => .done (.error "tokenizer limit reached")
      | _ + 1, [] => .done (.ok (tokens.reverse, pos))
      | fuel + 1, c :: cs =>
          let width := utf8ByteLength c
          if isSpace c then lexAsk fuel cs (pos + width) tokens
          else if c = '\n' then
            lexAsk fuel cs (pos + width) ({ kind := .newline, span := ⟨pos, pos + width⟩ } :: tokens)
          else if c = '/' then
            match cs with
            | [] => lexAsk fuel cs (pos + width) ({ kind := .other c, span := ⟨pos, pos + width⟩ } :: tokens)
            | second :: tail =>
                if second.toNat == '/'.toNat then
                  match skipComment tail (pos + 2) with
                  | (rest, next) => lexAsk fuel rest next tokens
                else lexAsk fuel cs (pos + width) ({ kind := .other c, span := ⟨pos, pos + width⟩ } :: tokens)
          else if isAlpha c then
            match takeWhile isAlphaNum cs with
            | (tail, rest) =>
                let scanned := c :: tail
                let stop := pos + utf8ByteLengthList scanned
                lexAsk fuel rest stop ({ kind := .ident (String.ofList scanned), span := ⟨pos, stop⟩ } :: tokens)
          else if isDigit c then
            match takeWhile isDigit cs with
            | (tail, rest) =>
                let scanned := c :: tail
                let stop := pos + utf8ByteLengthList scanned
                lexAsk fuel rest stop ({ kind := .numeral (digitsValue scanned), span := ⟨pos, stop⟩ } :: tokens)
          else
            if c.toNat == '<'.toNat then
              match cs with
              | [] => lexAsk fuel cs (pos + width) ({ kind := .lt, span := ⟨pos, pos + width⟩ } :: tokens)
              | second :: rest =>
                  if second.toNat == '='.toNat then
                    lexAsk fuel rest (pos + 2) ({ kind := .le, span := ⟨pos, pos + 2⟩ } :: tokens)
                  else lexAsk fuel cs (pos + width) ({ kind := .lt, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '>'.toNat then
              match cs with
              | [] => lexAsk fuel cs (pos + width) ({ kind := .gt, span := ⟨pos, pos + width⟩ } :: tokens)
              | second :: rest =>
                  if second.toNat == '='.toNat then
                    lexAsk fuel rest (pos + 2) ({ kind := .ge, span := ⟨pos, pos + 2⟩ } :: tokens)
                  else lexAsk fuel cs (pos + width) ({ kind := .gt, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '('.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .lparen, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == ')'.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .rparen, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '{'.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .lbrace, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '}'.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .rbrace, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '['.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .lbracket, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == ']'.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .rbracket, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == ','.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .comma, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == ';'.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .semi, span := ⟨pos, pos + width⟩ } :: tokens)
            else if c.toNat == '='.toNat then
              lexAsk fuel cs (pos + width) ({ kind := .assign, span := ⟨pos, pos + width⟩ } :: tokens)
            else lexAsk fuel cs (pos + width) ({ kind := .other c, span := ⟨pos, pos + width⟩ } :: tokens)

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .chars bytes => rfl
  | .tokenize bytes => by
      simp only [Slang.Derivation.Lexing.evaluate, Slang.Derivation.Lexing.body,
        Slang.Derivation.Lexing.lexAsk, Decode.tokenize, Slang.Derivation.fold_ask]
      cases result : Slang.Decode.decodeUtf8Chars bytes.data.toList with
      | error message => rfl
      | ok chars => rfl
  | .lex fuel chars pos tokens => by
      cases fuel with
      | zero =>
          rw [Slang.Derivation.Lexing.evaluate, Slang.Derivation.Lexing.body]
          rfl
      | succ fuel =>
        cases chars with
        | nil =>
            rw [Slang.Derivation.Lexing.evaluate, Slang.Derivation.Lexing.body]
            rfl
        | cons c cs =>
          cases cs with
          | nil =>
              simp only [Slang.Derivation.Lexing.evaluate, Slang.Derivation.Lexing.body,
                Slang.Derivation.Lexing.lexAsk, Slang.Decode.lexChars]
              repeat first
                | rw [Slang.Derivation.fold_ite]
                | rw [Slang.Derivation.fold_pair]
                | rw [Slang.Derivation.fold_ask]
                | rw [Slang.Derivation.fold_done]
                | rw [Slang.Derivation.Lexing.evaluate]
          | cons second tail =>
              simp only [Slang.Derivation.Lexing.evaluate, Slang.Derivation.Lexing.body,
                Slang.Derivation.Lexing.lexAsk, Slang.Decode.lexChars]
              repeat first
                | rw [Slang.Derivation.fold_ite]
                | rw [Slang.Derivation.fold_pair]
                | rw [Slang.Derivation.fold_ask]
                | rw [Slang.Derivation.fold_done]
                | rw [Slang.Derivation.Lexing.evaluate]

/-- A checked derivation of the real tokenizer's result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

#audit_empty Slang.Derivation.Lexing.evaluate : (q : Slang.Derivation.Lexing.Request) → Slang.Derivation.Lexing.Answer q
#audit_empty Slang.Derivation.Lexing.lexAsk : Nat → List Char → Nat → List Slang.Decode.Token → Slang.Derivation.Layer Slang.Derivation.Lexing.Request Slang.Derivation.Lexing.Answer (Slang.Decode.Result (List Slang.Decode.Token × Nat))
#audit_empty Slang.Derivation.Lexing.body_eq : ∀ (q : Slang.Derivation.Lexing.Request), Slang.Derivation.Lexing.evaluate q = Slang.Derivation.fold Slang.Derivation.Lexing.evaluate (Slang.Derivation.Lexing.body q)
#audit_empty Slang.Derivation.Lexing.Run.sound : ∀ {q : Slang.Derivation.Lexing.Request} {answer : Slang.Derivation.Lexing.Answer q}, Slang.Derivation.Lexing.Run q answer → Slang.Derivation.Lexing.evaluate q = answer

/-- Every scan request has a checked derivation of the real tokenizer's answer. -/
theorem lexChars_run : ∀ (fuel : Nat) (chars : List Char) (pos : Nat) (tokens : List Token),
    Run (.lex fuel chars pos tokens) (Decode.lexChars fuel chars pos tokens)
  | 0, _, _, _ => Slang.Derivation.Derives.done _
  | _ + 1, [], _, _ => Slang.Derivation.Derives.done _
  | fuel + 1, c :: cs, pos, tokens => by
      unfold Slang.Derivation.Lexing.Run Slang.Derivation.Lexing.body
        Slang.Derivation.Lexing.lexAsk Slang.Decode.lexChars
      dsimp only []
      by_cases h1 : isSpace c = true
      · simp only [if_pos h1]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
      · simp only [if_neg h1]
        by_cases h2 : c = '\n'
        · simp only [if_pos h2]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
        · simp only [if_neg h2]
          by_cases h3 : c = '/'
          · simp only [if_pos h3]
            cases cs with
            | nil => dsimp only []; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
            | cons second tail =>
                dsimp only []
                by_cases h4 : (second.toNat == '/'.toNat) = true
                · simp only [if_pos h4]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                · simp only [if_neg h4]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
          · simp only [if_neg h3]
            by_cases h5 : isAlpha c = true
            · simp only [if_pos h5]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
            · simp only [if_neg h5]
              by_cases h6 : isDigit c = true
              · simp only [if_pos h6]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
              · simp only [if_neg h6]
                by_cases h7 : (c.toNat == '<'.toNat) = true
                · simp only [if_pos h7]
                  cases cs with
                  | nil => dsimp only []; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                  | cons second rest =>
                      dsimp only []
                      by_cases h8 : (second.toNat == '='.toNat) = true
                      · simp only [if_pos h8]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                      · simp only [if_neg h8]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                · simp only [if_neg h7]
                  by_cases h9 : (c.toNat == '>'.toNat) = true
                  · simp only [if_pos h9]
                    cases cs with
                    | nil => dsimp only []; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                    | cons second rest =>
                        dsimp only []
                        by_cases h10 : (second.toNat == '='.toNat) = true
                        · simp only [if_pos h10]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                        · simp only [if_neg h10]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                  · simp only [if_neg h9]
                    by_cases h11 : (c.toNat == '('.toNat) = true
                    · simp only [if_pos h11]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                    · simp only [if_neg h11]
                      by_cases h12 : (c.toNat == ')'.toNat) = true
                      · simp only [if_pos h12]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                      · simp only [if_neg h12]
                        by_cases h13 : (c.toNat == '{'.toNat) = true
                        · simp only [if_pos h13]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                        · simp only [if_neg h13]
                          by_cases h14 : (c.toNat == '}'.toNat) = true
                          · simp only [if_pos h14]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                          · simp only [if_neg h14]
                            by_cases h15 : (c.toNat == '['.toNat) = true
                            · simp only [if_pos h15]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                            · simp only [if_neg h15]
                              by_cases h16 : (c.toNat == ']'.toNat) = true
                              · simp only [if_pos h16]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                              · simp only [if_neg h16]
                                by_cases h17 : (c.toNat == ','.toNat) = true
                                · simp only [if_pos h17]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                                · simp only [if_neg h17]
                                  by_cases h18 : (c.toNat == ';'.toNat) = true
                                  · simp only [if_pos h18]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                                  · simp only [if_neg h18]
                                    by_cases h19 : (c.toNat == '='.toNat) = true
                                    · simp only [if_pos h19]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)
                                    · simp only [if_neg h19]; exact Slang.Derivation.Derives.ask_done (lexChars_run fuel _ _ _)

/-- Every tokenize request has a checked derivation of the real tokenizer's answer. -/
theorem tokenize_run (bytes : ByteArray) :
    Run (.tokenize bytes) (Decode.tokenize bytes) := by
  unfold Slang.Derivation.Lexing.Run Slang.Derivation.Lexing.body
    Slang.Derivation.Lexing.lexAsk Slang.Decode.tokenize
  dsimp only []
  refine Slang.Derivation.Derives.ask (body := Slang.Derivation.Lexing.body)
    (request := .chars bytes.data.toList)
    (next := fun result => match result with
      | .ok chars => Slang.Derivation.Lexing.lexAsk (chars.length + 1) chars 0 []
      | .error message => .done (.error message))
    (answer := Decode.decodeUtf8Chars bytes.data.toList)
    (Slang.Derivation.Derives.done _) ?_
  cases h : Decode.decodeUtf8Chars bytes.data.toList with
  | error message =>
      dsimp only []
      exact Slang.Derivation.Derives.done _
  | ok chars =>
      dsimp only []
      exact Slang.Derivation.Derives.ask_done (body := Slang.Derivation.Lexing.body)
        (lexChars_run (chars.length + 1) chars 0 [])

#audit_empty Slang.Derivation.Lexing.lexChars_run : ∀ (fuel : Nat) (chars : List Char) (pos : Nat) (tokens : List Slang.Decode.Token), Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.lex fuel chars pos tokens) (Slang.Decode.lexChars fuel chars pos tokens)
#audit_empty Slang.Derivation.Lexing.tokenize_run : ∀ (bytes : ByteArray), Slang.Derivation.Lexing.Run (Slang.Derivation.Lexing.Request.tokenize bytes) (Slang.Decode.tokenize bytes)

end Slang.Derivation.Lexing
