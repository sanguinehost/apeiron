import Slang.Derivation.Steps
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 4096

namespace Slang.Derivation.Utilities

open Slang Slang.Decode

/-- Requests name the recursive list utilities of `Slang.Decode` instead of
evaluating them: `takeWhile`, `skipComment` and `dropNewlines`. Each request
carries exactly the arguments of the real function it names. -/
inductive Request where
  | takeWhile (p : Char → Bool) (chars : List Char)
  | skipComment (chars : List Char) (pos : Nat)
  | dropNewlines (tokens : List Token)

/-- Answer type per request, matching the real function's result type. -/
abbrev Answer : Request → Type
  | .takeWhile _ _ => List Char × List Char
  | .skipComment _ _ => List Char × Nat
  | .dropNewlines _ => List Token

/-- The real implementation, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .takeWhile p chars => Decode.takeWhile p chars
  | .skipComment chars pos => Decode.skipComment chars pos
  | .dropNewlines tokens => Decode.dropNewlines tokens

/-- One unfolding of each utility, exposing recursive calls as requests.

`takeWhile` and `skipComment` recurse on the head character; `dropNewlines`
recurses on the token list. The arms are explicit so no wildcard match is
emitted. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .takeWhile p chars =>
      match chars with
      | [] => .done ([], [])
      | c :: cs =>
          if p c then
            .ask (.takeWhile p cs) (fun answer => .done (c :: answer.1, answer.2))
          else .done ([], c :: cs)
  | .skipComment chars pos =>
      match chars with
      | [] => .done ([], pos)
      | c :: cs =>
          if c = '\n' then .done ('\n' :: cs, pos)
          else .ask (.skipComment cs (pos + Decode.utf8ByteLength c))
            (fun answer => .done answer)
  | .dropNewlines tokens =>
      match tokens with
      | [] => .done []
      | token :: rest =>
          if token.kind = .newline then
            .ask (.dropNewlines rest) (fun answer => .done answer)
          else .done (token :: rest)

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .takeWhile p chars => by
      cases chars with
      | nil => rfl
      | cons c cs =>
          dsimp only [evaluate, body, fold, Decode.takeWhile]
          by_cases hp : p c
          · conv => lhs; rw [if_pos hp]
            conv => rhs; rw [if_pos hp]
            rw [Slang.Derivation.fold_ask]
            rfl
          · conv => lhs; rw [if_neg hp]
            conv => rhs; rw [if_neg hp]
            rfl
  | .skipComment chars pos => by
      cases chars with
      | nil =>
          dsimp only [evaluate, body, fold]
          rw [Decode.skipComment.eq_1]
      | cons c cs =>
          dsimp only [evaluate, body]
          by_cases hnl : c = '\n'
          · subst hnl
            conv => lhs; rw [Decode.skipComment.eq_2]
            conv => rhs; rw [if_pos rfl]
            rw [Slang.Derivation.fold_done]
          · conv => lhs; rw [Decode.skipComment.eq_3 pos c cs hnl]
            conv => rhs; rw [if_neg hnl]
            rw [Slang.Derivation.fold_ask]
            dsimp only [evaluate]
            rw [Slang.Derivation.fold_done]
  | .dropNewlines tokens => by
      cases tokens with
      | nil => rfl
      | cons token rest =>
          dsimp only [evaluate, body, fold, Decode.dropNewlines]
          by_cases hn : token.kind = .newline
          · conv => lhs; rw [if_pos hn]
            conv => rhs; rw [if_pos hn]
            rw [Slang.Derivation.fold_ask]
            rfl
          · conv => lhs; rw [if_neg hn]
            conv => rhs; rw [if_neg hn]
            rfl

/-- A checked derivation of the real utility's result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

/-- Every `takeWhile` request has a checked derivation of the real result. -/
theorem takeWhile_run : ∀ (p : Char → Bool) (chars : List Char),
    Run (.takeWhile p chars) (Decode.takeWhile p chars)
  | p, [] => Derives.done _
  | p, c :: cs => by
      unfold Slang.Derivation.Utilities.Run Slang.Derivation.Utilities.body
      dsimp only [Decode.takeWhile]
      by_cases hp : p c
      · conv => arg 2; rw [if_pos hp]
        conv => arg 3; rw [if_pos hp]
        exact Derives.ask (body := body) (called := takeWhile_run p cs) (Derives.done _)
      · conv => arg 2; rw [if_neg hp]
        conv => arg 3; rw [if_neg hp]
        exact Derives.done _

/-- Every `skipComment` request has a checked derivation of the real result. -/
theorem skipComment_run : ∀ (chars : List Char) (pos : Nat),
    Run (.skipComment chars pos) (Decode.skipComment chars pos)
  | [], pos => Derives.done _
  | c :: cs, pos => by
      unfold Slang.Derivation.Utilities.Run Slang.Derivation.Utilities.body
      dsimp only []
      by_cases hnl : c = '\n'
      · subst hnl
        conv => arg 2; rw [if_pos rfl]
        conv => arg 3; rw [Decode.skipComment.eq_2]
        exact Derives.done _
      · conv => arg 2; rw [if_neg hnl]
        conv => arg 3; rw [Decode.skipComment.eq_3 pos c cs hnl]
        exact Derives.ask_done (body := body)
          (skipComment_run cs (pos + Decode.utf8ByteLength c))

/-- Every `dropNewlines` request has a checked derivation of the real result. -/
theorem dropNewlines_run : ∀ (tokens : List Token),
    Run (.dropNewlines tokens) (Decode.dropNewlines tokens)
  | [] => Derives.done _
  | token :: rest => by
      unfold Slang.Derivation.Utilities.Run Slang.Derivation.Utilities.body
      dsimp only [Decode.dropNewlines]
      by_cases hn : token.kind = .newline
      · conv => arg 2; rw [if_pos hn]
        conv => arg 3; rw [if_pos hn]
        exact Derives.ask_done (body := body) (dropNewlines_run rest)
      · conv => arg 2; rw [if_neg hn]
        conv => arg 3; rw [if_neg hn]
        exact Derives.done _

end Slang.Derivation.Utilities
