import Iota.Source.Tokenizer.Helpers

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.Tokenizer

open Iota Iota.Decode

/-- One successful lexer step, exposing every guard and maximal helper scan. -/
inductive LexHead : List Char → Nat → List Char → Nat → Option Token → Prop where
  | space (c : Char) (cs : List Char) (pos : Nat) (isSpace : Decode.isSpace c = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c) none
  | newline (cs : List Char) (pos : Nat) (notSpace : Decode.isSpace '\n' = false) :
      LexHead ('\n' :: cs) pos cs (pos + Decode.utf8ByteLength '\n')
        (some { kind := .newline, span := ⟨pos, pos + Decode.utf8ByteLength '\n'⟩ })
  | slashEnd (pos : Nat) (notSpace : Decode.isSpace '/' = false) (notNewline : '/' ≠ '\n') :
      LexHead ['/'] pos [] (pos + Decode.utf8ByteLength '/')
        (some { kind := .other '/', span := ⟨pos, pos + Decode.utf8ByteLength '/'⟩ })
  | comment (second : Char) (tail rest : List Char) (pos next : Nat)
      (notSpace : Decode.isSpace '/' = false) (notNewline : '/' ≠ '\n')
      (slashes : BEq.beq second.toNat '/'.toNat = true)
      (scan : SkipCommentScan tail (pos + 2) rest next) :
      LexHead ('/' :: second :: tail) pos rest next none
  | slash (second : Char) (tail : List Char) (pos : Nat)
      (notSpace : Decode.isSpace '/' = false) (notNewline : '/' ≠ '\n')
      (notSlashes : BEq.beq second.toNat '/'.toNat = false) :
      LexHead ('/' :: second :: tail) pos (second :: tail) (pos + Decode.utf8ByteLength '/')
        (some { kind := .other '/', span := ⟨pos, pos + Decode.utf8ByteLength '/'⟩ })
  | ident (c : Char) (cs front rest chars : List Char) (pos size stop : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (alpha : Decode.isAlpha c = true) (run : TakeWhileScan Decode.isAlphaNum cs front rest)
      (charsEq : chars = c :: front) (length : Utf8LengthFold 0 chars size)
      (stopEq : stop = pos + size) :
      LexHead (c :: cs) pos rest stop
        (some { kind := .ident (String.ofList chars), span := ⟨pos, stop⟩ })
  | numeral (c : Char) (cs front rest chars : List Char) (pos size stop value : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (digit : Decode.isDigit c = true)
      (run : TakeWhileScan Decode.isDigit cs front rest) (charsEq : chars = c :: front)
      (length : Utf8LengthFold 0 chars size) (stopEq : stop = pos + size)
      (digits : DigitsFold 0 chars value) :
      LexHead (c :: cs) pos rest stop
        (some { kind := .numeral value, span := ⟨pos, stop⟩ })
  | ltEnd (pos : Nat) (notSpace : Decode.isSpace '<' = false) (notNewline : '<' ≠ '\n')
      (notSlash : '<' ≠ '/') (notAlpha : Decode.isAlpha '<' = false)
      (notDigit : Decode.isDigit '<' = false) (isLt : BEq.beq '<'.toNat '<'.toNat = true) :
      LexHead ['<'] pos [] (pos + Decode.utf8ByteLength '<')
        (some { kind := .lt, span := ⟨pos, pos + Decode.utf8ByteLength '<'⟩ })
  | le (second : Char) (rest : List Char) (pos : Nat)
      (notSpace : Decode.isSpace '<' = false) (notNewline : '<' ≠ '\n')
      (notSlash : '<' ≠ '/') (notAlpha : Decode.isAlpha '<' = false)
      (notDigit : Decode.isDigit '<' = false) (isLt : BEq.beq '<'.toNat '<'.toNat = true)
      (isEqual : BEq.beq second.toNat '='.toNat = true) :
      LexHead ('<' :: second :: rest) pos rest (pos + 2)
        (some { kind := .le, span := ⟨pos, pos + 2⟩ })
  | lt (second : Char) (rest : List Char) (pos : Nat)
      (notSpace : Decode.isSpace '<' = false) (notNewline : '<' ≠ '\n')
      (notSlash : '<' ≠ '/') (notAlpha : Decode.isAlpha '<' = false)
      (notDigit : Decode.isDigit '<' = false) (isLt : BEq.beq '<'.toNat '<'.toNat = true)
      (notEqual : BEq.beq second.toNat '='.toNat = false) :
      LexHead ('<' :: second :: rest) pos (second :: rest) (pos + Decode.utf8ByteLength '<')
        (some { kind := .lt, span := ⟨pos, pos + Decode.utf8ByteLength '<'⟩ })
  | gtEnd (pos : Nat) (notSpace : Decode.isSpace '>' = false) (notNewline : '>' ≠ '\n')
      (notSlash : '>' ≠ '/') (notAlpha : Decode.isAlpha '>' = false)
      (notDigit : Decode.isDigit '>' = false) (notLt : BEq.beq '>'.toNat '<'.toNat = false)
      (isGt : BEq.beq '>'.toNat '>'.toNat = true) :
      LexHead ['>'] pos [] (pos + Decode.utf8ByteLength '>')
        (some { kind := .gt, span := ⟨pos, pos + Decode.utf8ByteLength '>'⟩ })
  | ge (second : Char) (rest : List Char) (pos : Nat)
      (notSpace : Decode.isSpace '>' = false) (notNewline : '>' ≠ '\n')
      (notSlash : '>' ≠ '/') (notAlpha : Decode.isAlpha '>' = false)
      (notDigit : Decode.isDigit '>' = false) (notLt : BEq.beq '>'.toNat '<'.toNat = false)
      (isGt : BEq.beq '>'.toNat '>'.toNat = true) (isEqual : BEq.beq second.toNat '='.toNat = true) :
      LexHead ('>' :: second :: rest) pos rest (pos + 2)
        (some { kind := .ge, span := ⟨pos, pos + 2⟩ })
  | gt (second : Char) (rest : List Char) (pos : Nat)
      (notSpace : Decode.isSpace '>' = false) (notNewline : '>' ≠ '\n')
      (notSlash : '>' ≠ '/') (notAlpha : Decode.isAlpha '>' = false)
      (notDigit : Decode.isDigit '>' = false) (notLt : BEq.beq '>'.toNat '<'.toNat = false)
      (isGt : BEq.beq '>'.toNat '>'.toNat = true) (notEqual : BEq.beq second.toNat '='.toNat = false) :
      LexHead ('>' :: second :: rest) pos (second :: rest) (pos + Decode.utf8ByteLength '>')
        (some { kind := .gt, span := ⟨pos, pos + Decode.utf8ByteLength '>'⟩ })
  | lparen (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (guard : BEq.beq c.toNat '('.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .lparen, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | rparen (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (guard : BEq.beq c.toNat ')'.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .rparen, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | lbrace (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (guard : BEq.beq c.toNat '{'.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .lbrace, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | rbrace (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (guard : BEq.beq c.toNat '}'.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .rbrace, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | lbracket (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (guard : BEq.beq c.toNat '['.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .lbracket, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | rbracket (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (notLbracket : BEq.beq c.toNat '['.toNat = false) (guard : BEq.beq c.toNat ']'.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .rbracket, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | comma (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (notLbracket : BEq.beq c.toNat '['.toNat = false) (notRbracket : BEq.beq c.toNat ']'.toNat = false)
      (guard : BEq.beq c.toNat ','.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .comma, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | semi (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (notLbracket : BEq.beq c.toNat '['.toNat = false) (notRbracket : BEq.beq c.toNat ']'.toNat = false)
      (notComma : BEq.beq c.toNat ','.toNat = false) (guard : BEq.beq c.toNat ';'.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .semi, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | assign (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (notLbracket : BEq.beq c.toNat '['.toNat = false) (notRbracket : BEq.beq c.toNat ']'.toNat = false)
      (notComma : BEq.beq c.toNat ','.toNat = false) (notSemi : BEq.beq c.toNat ';'.toNat = false)
      (guard : BEq.beq c.toNat '='.toNat = true) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .assign, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })
  | other (c : Char) (cs : List Char) (pos : Nat)
      (notSpace : Decode.isSpace c = false) (notNewline : c ≠ '\n') (notSlash : c ≠ '/')
      (notAlpha : Decode.isAlpha c = false) (notDigit : Decode.isDigit c = false)
      (notLt : BEq.beq c.toNat '<'.toNat = false) (notGt : BEq.beq c.toNat '>'.toNat = false)
      (notLparen : BEq.beq c.toNat '('.toNat = false) (notRparen : BEq.beq c.toNat ')'.toNat = false)
      (notLbrace : BEq.beq c.toNat '{'.toNat = false) (notRbrace : BEq.beq c.toNat '}'.toNat = false)
      (notLbracket : BEq.beq c.toNat '['.toNat = false) (notRbracket : BEq.beq c.toNat ']'.toNat = false)
      (notComma : BEq.beq c.toNat ','.toNat = false) (notSemi : BEq.beq c.toNat ';'.toNat = false)
      (notAssign : BEq.beq c.toNat '='.toNat = false) :
      LexHead (c :: cs) pos cs (pos + Decode.utf8ByteLength c)
        (some { kind := .other c, span := ⟨pos, pos + Decode.utf8ByteLength c⟩ })

namespace LexHead

/-- A structural lexer head and successful recursive tail compose to one `lexChars` step. -/
theorem sound {chars rest : List Char} {pos next : Nat} {emitted : Option Token}
    (head : LexHead chars pos rest next emitted) {n : Nat} {reversed : List Token}
    {result : List Token × Nat}
    (tail : Decode.lexChars n rest next (match emitted with | none => reversed | some token => token :: reversed) = .ok result) :
    Decode.lexChars (n + 1) chars pos reversed = .ok result := by
  have boolNeg : ∀ {b : Bool}, b = false → b ≠ true := by
    intro b h
    rw [h]
    exact Bool.false_ne_true
  cases head with
  | space c cs pos isSpace =>
      unfold Decode.lexChars
      rw [if_pos isSpace]
      dsimp only at tail
      exact tail
  | newline cs pos notSpace =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_pos rfl]
      dsimp only at tail
      exact tail
  | slashEnd pos notSpace notNewline =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_pos rfl]
      dsimp only at tail ⊢
      exact tail
  | comment second body rest pos next notSpace notNewline slashes scan =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_pos rfl]
      dsimp only
      rw [if_pos slashes, scan.sound]
      dsimp only at tail
      exact tail
  | slash second body pos notSpace notNewline notSlashes =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_pos rfl]
      dsimp only
      rw [if_neg (boolNeg notSlashes)]
      dsimp only at tail ⊢
      exact tail
  | ident c cs front rest chars pos size stop notSpace notNewline notSlash alpha run charsEq length stopEq =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash, if_pos alpha, run.sound]
      dsimp only
      rw [← charsEq]
      unfold Decode.utf8ByteLengthList
      rw [length.sound, ← stopEq]
      dsimp only at tail
      exact tail
  | numeral c cs front rest chars pos size stop value notSpace notNewline notSlash notAlpha digit run charsEq length stopEq digits =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_pos digit, run.sound]
      dsimp only
      rw [← charsEq]
      unfold Decode.utf8ByteLengthList Decode.digitsValue
      rw [length.sound, digits.sound, ← stopEq]
      dsimp only at tail
      exact tail
  | ltEnd pos notSpace notNewline notSlash notAlpha notDigit isLt =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_pos isLt]
      dsimp only at tail ⊢
      exact tail
  | le second rest pos notSpace notNewline notSlash notAlpha notDigit isLt isEqual =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_pos isLt]
      dsimp only
      rw [if_pos isEqual]
      dsimp only at tail
      exact tail
  | lt second rest pos notSpace notNewline notSlash notAlpha notDigit isLt notEqual =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_pos isLt]
      dsimp only
      rw [if_neg (boolNeg notEqual)]
      dsimp only at tail ⊢
      exact tail
  | gtEnd pos notSpace notNewline notSlash notAlpha notDigit notLt isGt =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_neg (boolNeg notLt),
        if_pos isGt]
      dsimp only at tail ⊢
      exact tail
  | ge second rest pos notSpace notNewline notSlash notAlpha notDigit notLt isGt isEqual =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_neg (boolNeg notLt),
        if_pos isGt]
      dsimp only
      rw [if_pos isEqual]
      dsimp only at tail
      exact tail
  | gt second rest pos notSpace notNewline notSlash notAlpha notDigit notLt isGt notEqual =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg notSpace), if_neg notNewline, if_neg notSlash,
        if_neg (boolNeg notAlpha), if_neg (boolNeg notDigit), if_neg (boolNeg notLt),
        if_pos isGt]
      dsimp only
      rw [if_neg (boolNeg notEqual)]
      dsimp only at tail ⊢
      exact tail
  | lparen c cs pos h1 h2 h3 h4 h5 h6 h7 h8 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7), if_pos h8]
      dsimp only at tail ⊢
      exact tail
  | rparen c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_pos h9]
      dsimp only at tail ⊢
      exact tail
  | lbrace c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_pos h10]
      dsimp only at tail ⊢
      exact tail
  | rbrace c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10), if_pos h11]
      dsimp only at tail ⊢
      exact tail
  | lbracket c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_pos h12]
      dsimp only at tail ⊢
      exact tail
  | rbracket c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_neg (boolNeg h12), if_pos h13]
      dsimp only at tail ⊢
      exact tail
  | comma c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_neg (boolNeg h12), if_neg (boolNeg h13), if_pos h14]
      dsimp only at tail ⊢
      exact tail
  | semi c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_neg (boolNeg h12), if_neg (boolNeg h13),
        if_neg (boolNeg h14), if_pos h15]
      dsimp only at tail ⊢
      exact tail
  | assign c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_neg (boolNeg h12), if_neg (boolNeg h13),
        if_neg (boolNeg h14), if_neg (boolNeg h15), if_pos h16]
      dsimp only at tail ⊢
      exact tail
  | other c cs pos h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 =>
      unfold Decode.lexChars
      rw [if_neg (boolNeg h1), if_neg h2, if_neg h3, if_neg (boolNeg h4),
        if_neg (boolNeg h5), if_neg (boolNeg h6), if_neg (boolNeg h7),
        if_neg (boolNeg h8), if_neg (boolNeg h9), if_neg (boolNeg h10),
        if_neg (boolNeg h11), if_neg (boolNeg h12), if_neg (boolNeg h13),
        if_neg (boolNeg h14), if_neg (boolNeg h15), if_neg (boolNeg h16)]
      dsimp only at tail ⊢
      exact tail

end LexHead

/-- Constructive reversal of the lexer accumulator. -/
inductive ReverseTokens : List Token → List Token → Prop where
  | result (reversed : List Token) : ReverseTokens reversed reversed.reverse

namespace ReverseTokens

theorem of_eq {reversed tokens : List Token} (h : reversed.reverse = tokens) :
    ReverseTokens reversed tokens := by
  rw [← h]
  exact ReverseTokens.result reversed

/-- A structural accumulator reversal entails `List.reverse`. -/
theorem sound {reversed tokens : List Token} (scan : ReverseTokens reversed tokens) :
    reversed.reverse = tokens := by
  cases scan
  rfl

end ReverseTokens

/-- A successful structural character scan mirrors `Decode.lexChars`. -/
inductive CharsScan : Nat → List Char → Nat → List Token → List Token → Nat → Prop where
  | done {n pos : Nat} {reversed tokens : List Token} (reverse : ReverseTokens reversed tokens) :
      CharsScan (n + 1) [] pos reversed tokens pos
  | nextNone {n pos next consumed : Nat} {chars rest : List Char} {reversed tokens : List Token}
      (head : LexHead chars pos rest next none)
      (tail : CharsScan n rest next reversed tokens consumed) :
      CharsScan (n + 1) chars pos reversed tokens consumed
  | nextToken {n pos next consumed : Nat} {chars rest : List Char}
      {reversed tokens : List Token} {token : Token}
      (head : LexHead chars pos rest next (some token))
      (tail : CharsScan n rest next (token :: reversed) tokens consumed) :
      CharsScan (n + 1) chars pos reversed tokens consumed

namespace CharsScan

theorem «continue» {fuel0 fuel1 pos0 pos1 consumed : Nat} {chars0 chars1 : List Char}
    {reversed0 reversed1 tokens : List Token}
    («prefix» : CharsScan fuel1 chars1 pos1 reversed1 tokens consumed →
      CharsScan fuel0 chars0 pos0 reversed0 tokens consumed)
    (tail : CharsScan fuel1 chars1 pos1 reversed1 tokens consumed) :
    CharsScan fuel0 chars0 pos0 reversed0 tokens consumed :=
  «prefix» tail

/-- A structural character scan entails the successful executable lexer result. -/
theorem sound {n pos consumed : Nat} {chars : List Char} {reversed tokens : List Token}
    (scan : CharsScan n chars pos reversed tokens consumed) :
    Decode.lexChars n chars pos reversed = .ok (tokens, consumed) := by
  induction scan with
  | done reverse =>
      unfold Decode.lexChars
      rw [reverse.sound]
  | nextNone head _ tailSound => exact head.sound tailSound
  | nextToken head _ tailSound => exact head.sound tailSound

end CharsScan

end Iota.Source.Tokenizer
