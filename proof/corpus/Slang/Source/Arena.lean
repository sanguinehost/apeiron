import Slang.Source.Bytes
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

/-- The sublist `values[start..stop)`. -/
def slice {α : Type} (values : List α) (start stop : Nat) : List α :=
  (values.drop start).take (stop - start)

/-- A half-open interval `[start, stop)` checked against a bound. -/
structure CheckedInterval (bound : Nat) where
  start : Nat
  stop : Nat
  ordered : start ≤ stop
  bounded : stop ≤ bound

/-- The elements of `values` selected by a checked interval. -/
def CheckedInterval.contents {α : Type} {values : List α}
    (interval : CheckedInterval values.length) : List α :=
  slice values interval.start interval.stop

/-- One source file: its canonical path, its bytes and its decoded tokens. -/
structure ArenaFile where
  path : SourceBytes
  bytes : SourceBytes
  tokens : List Slang.Decode.Token

/-- A source arena: a list of files whose canonical paths are pairwise distinct. -/
structure SourceArena where
  files : List ArenaFile
  pathUnique : ∀ i j : Fin files.length, (files.get i).path = (files.get j).path → i = j

/-- An index into the arena's files. -/
abbrev FileId (arena : SourceArena) := Fin arena.files.length

/-- The file at `id`. -/
def SourceArena.file (arena : SourceArena) (id : FileId arena) : ArenaFile :=
  arena.files.get id

/-- Shift a token span so that `origin` becomes byte zero. -/
def rebaseToken (origin : Nat) (token : Slang.Decode.Token) : Slang.Decode.Token :=
  { token with
    span := { start := token.span.start - origin, stop := token.span.stop - origin } }

/-- A checked reference into one arena file: a byte interval whose contents tokenize exactly to the
tokens selected by the token interval, rebased to the byte interval's start. The `correspondence`
field is the checked tie between the two intervals; a copied token list without its owning byte
interval cannot supply it. -/
structure SourceRef (arena : SourceArena) where
  file : FileId arena
  byteInterval : CheckedInterval (arena.file file).bytes.length
  tokenInterval : CheckedInterval (arena.file file).tokens.length
  correspondence :
    Slang.Decode.tokenize (encodeBytes byteInterval.contents) =
      .ok ((tokenInterval.contents.map (rebaseToken byteInterval.start)),
        byteInterval.contents.length)

/-- The byte interval is ordered and lies inside the file's bytes. -/
theorem SourceRef.byte_in_bounds {arena : SourceArena} (ref : SourceRef arena) :
    ref.byteInterval.start ≤ ref.byteInterval.stop ∧
      ref.byteInterval.stop ≤ (arena.file ref.file).bytes.length :=
  ⟨ref.byteInterval.ordered, ref.byteInterval.bounded⟩

/-- The token interval is ordered and lies inside the file's tokens. -/
theorem SourceRef.token_in_bounds {arena : SourceArena} (ref : SourceRef arena) :
    ref.tokenInterval.start ≤ ref.tokenInterval.stop ∧
      ref.tokenInterval.stop ≤ (arena.file ref.file).tokens.length :=
  ⟨ref.tokenInterval.ordered, ref.tokenInterval.bounded⟩

/-- The token interval's tokens are exactly the tokenization of the byte interval's contents, rebased
to the byte interval's start. -/
theorem SourceRef.exact_tokens {arena : SourceArena} (ref : SourceRef arena) :
    Slang.Decode.tokenize (encodeBytes ref.byteInterval.contents) =
      .ok ((ref.tokenInterval.contents.map (rebaseToken ref.byteInterval.start)),
        ref.byteInterval.contents.length) :=
  ref.correspondence

/-! ### Rejecting fixture

The bytes `"ab"` tokenize to the single identifier token `ab`. A copied `newline` token list does not
own those bytes, so the `correspondence` equation a `SourceRef` carrying it would need is false. -/

private def sampleBytes : SourceBytes := [⟨97, by decide⟩, ⟨98, by decide⟩]

private def copiedTokens : List Slang.Decode.Token :=
  [{ kind := .newline, span := { start := 0, stop := 1 } }]

private def firstKind :
    Slang.Decode.Result (List Slang.Decode.Token × Nat) → Option Slang.Decode.TokenKind
  | .ok (tokens, _) => tokens.head?.map (fun token => token.kind)
  | .error _ => none

/-- A copied token list cannot be the `correspondence` witness for the owning byte interval. -/
theorem copiedTokens_rejected :
    ¬ (Slang.Decode.tokenize (encodeBytes sampleBytes) =
      .ok (copiedTokens, sampleBytes.length)) := by
  intro h
  exact absurd (congrArg firstKind h) (by decide)

/-- No `SourceRef` can carry the copied token list as the tokens of the `"ab"` byte interval: its
`correspondence` field would assert the equation `copiedTokens_rejected` refutes. -/
theorem copiedTokens_not_a_ref {arena : SourceArena} (ref : SourceRef arena)
    (hbytes : ref.byteInterval.contents = sampleBytes)
    (hstart : ref.byteInterval.start = 0)
    (htokens : ref.tokenInterval.contents = copiedTokens) : False := by
  have h := ref.exact_tokens
  rw [hbytes, htokens, hstart] at h
  exact copiedTokens_rejected h

#guard (match Slang.Decode.tokenize (encodeBytes sampleBytes) with
  | .ok (tokens, _) => tokens == copiedTokens
  | .error _ => false) == false

end Slang.Source
