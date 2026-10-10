import Iota.Source.Bytes
import Iota.Decode
import Iota.Refinement.Context

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source

/-- The sublist `values[start..stop)`. -/
-- Temporal O(stop) element visits: `List.drop start` walks `start` cons cells and `List.take
-- (stop - start)` walks the next `stop - start`, one O(1) step per cell; the recursion closes on the
-- finite list. Spatial O(stop - start) cons cells copied by `take`; `drop` shares the tail.
def slice {α : Type} (values : List α) (start stop : Nat) : List α :=
  (values.drop start).take (stop - start)

/-- A half-open interval `[start, stop)` checked against a bound. -/
structure CheckedInterval (bound : Nat) where
  start : Nat
  stop : Nat
  ordered : start ≤ stop
  bounded : stop ≤ bound

/-- The elements of `values` selected by a checked interval. -/
-- Cost is that of `slice values interval.start interval.stop`: temporal O(interval.stop) element
-- visits and spatial O(interval.stop - interval.start) copied cons cells, since `bounded` gives
-- `interval.stop ≤ values.length`.
def CheckedInterval.contents {α : Type} {values : List α}
    (interval : CheckedInterval values.length) : List α :=
  slice values interval.start interval.stop

/-- One source file: its canonical path, its bytes and its decoded tokens. -/
structure ArenaFile where
  path : SourceBytes
  bytes : SourceBytes
  tokens : List Iota.Decode.Token

/-- A source arena: a list of files whose canonical paths are pairwise distinct. -/
structure SourceArena where
  files : List ArenaFile
  pathUnique : ∀ i j : Fin files.length, (files.get i).path = (files.get j).path → i = j

/-- An index into the arena's files. -/
abbrev FileId (arena : SourceArena) := Fin arena.files.length

/-- The file at `id`. -/
-- Temporal O(id.val) cons-cell steps: `List.get` recurses down `arena.files` to position `id.val`,
-- one step per cell, closing at the `Fin` bound. Spatial O(1): the recursion is tail-recursive.
def SourceArena.file (arena : SourceArena) (id : FileId arena) : ArenaFile :=
  arena.files.get id

/-- Shift a token span so that `origin` becomes byte zero. -/
def rebaseToken (origin : Nat) (token : Iota.Decode.Token) : Iota.Decode.Token :=
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
    Iota.Decode.tokenize (encodeBytes byteInterval.contents) =
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
    Iota.Decode.tokenize (encodeBytes ref.byteInterval.contents) =
      .ok ((ref.tokenInterval.contents.map (rebaseToken ref.byteInterval.start)),
        ref.byteInterval.contents.length) :=
  ref.correspondence

/-! ### Rejecting fixture

The bytes `"ab"` tokenize to the single identifier token `ab`. A copied `newline` token list does not
own those bytes, so the `correspondence` equation a `SourceRef` carrying it would need is false. -/

private def sampleBytes : SourceBytes := [⟨97, by decide⟩, ⟨98, by decide⟩]

private def copiedTokens : List Iota.Decode.Token :=
  [{ kind := .newline, span := { start := 0, stop := 1 } }]

private def firstKind :
    Iota.Decode.Result (List Iota.Decode.Token × Nat) → Option Iota.Decode.TokenKind
  | .ok (tokens, _) => tokens.head?.map (fun token => token.kind)
  | .error _ => none

/-- A copied token list cannot be the `correspondence` witness for the owning byte interval. -/
theorem copiedTokens_rejected :
    ¬ (Iota.Decode.tokenize (encodeBytes sampleBytes) =
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

#guard (match Iota.Decode.tokenize (encodeBytes sampleBytes) with
  | .ok (tokens, _) => tokens == copiedTokens
  | .error _ => false) == false

/-! ### Checked parsed source references

A bare `SourceRef` checks that a byte interval tokenizes to a token interval, but it does not carry
the decoded program. The memo value and the machine context need that program, and they must not
fall back to a `default` or an `Option.getD`. `ParsedRef` adds the decoded program and the checked
equation tying it to the referenced bytes, so the program is the decode of those bytes rather than a
supplied value. -/

/-- A checked reference together with the program its bytes decode to. The `decode_eq` field is the
tie: the program is exactly the `decodeTable` result for the referenced byte interval, so a
`ParsedRef` whose program disagrees with its bytes has no inhabitant. -/
structure ParsedRef (arena : SourceArena) where
  ref : SourceRef arena
  program : Iota.Program
  certificate : Iota.Decode.Certificate
  certificateValid : certificate.Valid
  decode_eq :
    Iota.Decode.decodeTable (encodeBytes ref.byteInterval.contents) =
      .ok ⟨Iota.Decode.Decoded.mk program certificate, certificateValid⟩

/-- The program is the checked decode of the referenced bytes. -/
theorem ParsedRef.exact_decode {arena : SourceArena} (parsed : ParsedRef arena) :
    Iota.Decode.decodeTable (encodeBytes parsed.ref.byteInterval.contents) =
      .ok ⟨Iota.Decode.Decoded.mk parsed.program parsed.certificate,
        parsed.certificateValid⟩ :=
  parsed.decode_eq

/-- Project the program out of a decode result, so equality of results yields equality of programs.
The `error` branch is unreachable in the uses below. -/
private def resultProgram :
    Iota.Decode.Result Iota.Decode.CertifiedDecoded → Iota.Program
  | .ok decoded => decoded.val.program
  | .error _ => ⟨[], [], 0, 0⟩

/-- The bytes determine the program: two parsed references over equal byte contents carry equal
programs. This is what forbids a `ParsedRef` from carrying a program that does not match its bytes. -/
theorem ParsedRef.program_determined {arena : SourceArena} (left right : ParsedRef arena)
    (hbytes : left.ref.byteInterval.contents = right.ref.byteInterval.contents) :
    left.program = right.program := by
  have hright : Iota.Decode.decodeTable (encodeBytes left.ref.byteInterval.contents) =
      .ok ⟨Iota.Decode.Decoded.mk right.program right.certificate,
        right.certificateValid⟩ := by
    rw [hbytes]
    exact right.decode_eq
  have hsame : (Except.ok
        ⟨Iota.Decode.Decoded.mk left.program left.certificate, left.certificateValid⟩ :
        Iota.Decode.Result Iota.Decode.CertifiedDecoded) =
      Except.ok ⟨Iota.Decode.Decoded.mk right.program right.certificate,
        right.certificateValid⟩ :=
    left.decode_eq.symm.trans hright
  exact congrArg resultProgram hsame

/-- A decode witness for the same bytes whose program differs from the checked program cannot
inhabit `ParsedRef`: the `decode_eq` field pins the program. -/
theorem ParsedRef.mismatched_program_rejected {arena : SourceArena} (parsed : ParsedRef arena)
    (wrong : Iota.Program) (certificate : Iota.Decode.Certificate)
    (valid : certificate.Valid)
    (hwrong : Iota.Decode.decodeTable (encodeBytes parsed.ref.byteInterval.contents) =
      .ok ⟨Iota.Decode.Decoded.mk wrong certificate, valid⟩)
    (hne : wrong ≠ parsed.program) : False := by
  have hsame : (Except.ok
        ⟨Iota.Decode.Decoded.mk parsed.program parsed.certificate, parsed.certificateValid⟩ :
        Iota.Decode.Result Iota.Decode.CertifiedDecoded) =
      Except.ok ⟨Iota.Decode.Decoded.mk wrong certificate, valid⟩ :=
    parsed.decode_eq.symm.trans hwrong
  have hprog : parsed.program = wrong := congrArg resultProgram hsame
  exact hne hprog.symm

/-- The machine context of a parsed reference. Its constants come from the checked program's
declarations and its functions from the checked program's functions; no `default` and no
`Option.getD` are used. -/
def ParsedRef.context {arena : SourceArena} (parsed : ParsedRef arena) : Iota.Machine.Context :=
  Iota.Refinement.checkedContext parsed.program

/-- The context's constant environment is the one derived from the checked program. -/
theorem ParsedRef.context_constants {arena : SourceArena} (parsed : ParsedRef arena) :
    parsed.context.constants = Iota.Refinement.constantEnvironment parsed.program :=
  Iota.Refinement.checkedContext_constants parsed.program

private def wrongProgram : Iota.Program :=
  { constants := [], functions := [], sourceBytes := 0, consumedBytes := 0 }

#guard (match Iota.Decode.decodeTable (encodeBytes sampleBytes) with
  | .ok decoded => decoded.val.program == wrongProgram
  | .error _ => false) == false

end Iota.Source
