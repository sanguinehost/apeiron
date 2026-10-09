import Apeiron.Linter.Notation

/-!
# A bounded parser for grouped literals

@name Grouped Expression Parser
@grade proved

The input `[lparen, lparen, nat 7, rparen, rparen]` has one meaning and one remainder: the tree
`lit 7` and the empty suffix. Parentheses disappear from the tree, but their closing tokens must
still be consumed. `Derives` records both facts independently of the executable parser.

This first grammar contains natural literals and grouping. Addition, multiplication, precedence,
Slang token arenas and production-parser refinement are outside the theorem statements in this
file.
-/

namespace Apeiron
namespace Parser

/-- Tokens in the grouped-literal grammar.

    Token                         [one input token]
    nat (n : ℕ)                   [a natural-number literal]
    lparen                        [an opening parenthesis]
    rparen                        [a closing parenthesis] -/
inductive Token where
  | nat (n : Nat)
  | lparen
  | rparen
  deriving DecidableEq, Repr

/-- Abstract syntax produced by the grouped-literal parser.

    Expr                          [one parsed expression]
    lit (n : ℕ)                  [a natural-number literal tree] -/
inductive Expr where
  | lit (n : Nat)
  deriving DecidableEq, Repr

/-- A parser result distinguishes success, syntax refusal and exhausted fuel.

    Result                        [one bounded parser result]
    ok (tree : Expr) (rest : List Token) [a tree and the exact unconsumed suffix]
    syntax                        [the input does not fit the grammar]
    fuel                          [the parser exhausted its explicit bound] -/
inductive Result where
  | ok (tree : Expr) (rest : List Token)
  | syntax
  | fuel
  deriving DecidableEq, Repr

/-- The grammar relation, including the exact unconsumed suffix.

    Derives input tree rest       [input derives tree and leaves exactly rest]

    `List` [a finite sequence of things, in order]
    `Expr` [an expression a law or spec row is allowed to state]
    `→` [if the left holds so does the right] -/
inductive Derives : List Token → Expr → List Token → Prop where
  | lit (n : Nat) (rest : List Token) : Derives (.nat n :: rest) (.lit n) rest
  | group {inside rest : List Token} {tree : Expr} :
      Derives inside tree (.rparen :: rest) → Derives (.lparen :: inside) tree rest

/-- Execute the grouped-literal parser under a finite descent budget.

    parse fuel input              [run with fuel on the input tokens]
    Result                        [success with suffix, syntax refusal, or fuel refusal]

    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
def parse : Nat → List Token → Result
  | 0, _ => .fuel
  | _ + 1, [] => .syntax
  | _ + 1, .nat n :: rest => .ok (.lit n) rest
  | fuel + 1, .lparen :: inside =>
      match parse fuel inside with
      | .ok tree (.rparen :: rest) => .ok tree rest
      | .fuel => .fuel
      | _ => .syntax
  | _ + 1, .rparen :: _ => .syntax

/-- A successful executable parse is a derivation of the same tree and exact suffix.

    parse fuel input = .ok tree rest      [the executable parser succeeds]
    Derives input tree rest               [the grammar derives that tree and suffix]

    `Expr` [an expression a law or spec row is allowed to state]
    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
theorem parse_sound {fuel : Nat} {input : List Token} {tree : Expr} {rest : List Token}
    (h : parse fuel input = .ok tree rest) : Derives input tree rest := by
  induction fuel generalizing input tree rest with
  | zero => simp [parse] at h
  | succ fuel ih =>
      cases input with
      | nil => simp [parse] at h
      | cons token input =>
          cases token with
          | nat n =>
              simp [parse] at h
              obtain ⟨rfl, rfl⟩ := h
              exact .lit n input
          | rparen => simp [parse] at h
          | lparen =>
              simp only [parse] at h
              cases hp : parse fuel input with
              | «syntax» => simp [hp] at h
              | fuel => simp [hp] at h
              | ok inner suffix =>
                  cases suffix with
                  | nil => simp [hp] at h
                  | cons close suffix =>
                      cases close with
                      | nat n => simp [hp] at h
                      | lparen => simp [hp] at h
                      | rparen =>
                          simp [hp] at h
                          obtain ⟨rfl, rfl⟩ := h
                          exact .group (ih hp)

/-- Zero fuel always gives the fuel refusal.

    (input : List Token)           [any input token sequence]
    parse 0 input = .fuel          [the explicit budget is exhausted] -/
theorem parse_zero_fuel (input : List Token) : parse 0 input = .fuel := rfl

/-- One grouped literal is inhabited and consumes its closing token exactly.

    parse 2 [.lparen, .nat 7, .rparen] = .ok (.lit 7) []
      [the parser returns literal seven and leaves no token] -/
theorem parse_grouped_literal :
    parse 2 [.lparen, .nat 7, .rparen] = .ok (.lit 7) [] := rfl

/-- Nested grouping preserves the literal tree and consumes both closing tokens.

    parse 3 [.lparen, .lparen, .nat 7, .rparen, .rparen] = .ok (.lit 7) []
      [the parser returns literal seven and leaves no token] -/
theorem parse_nested_grouped_literal :
    parse 3 [.lparen, .lparen, .nat 7, .rparen, .rparen] = .ok (.lit 7) [] := rfl

end Parser
end Apeiron
