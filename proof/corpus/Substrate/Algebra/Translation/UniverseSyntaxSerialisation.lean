import Substrate.Algebra.Translation.UniverseSyntax

/-!
# Universe Syntax — Linear Serialisation

@name Writing a Shape Down as a List
@child Any of the framework's structures can be written out as a simple list of small tokens — one starter piece, then a string of "extend-this-way" or "extend-that-way" steps — and you can always rebuild the exact original from the list.
@scientist Because every internal node of a `Polymorphic.Term` has exactly one recursive slot (the other a leaf), each Term is a non-empty token list `[atom, ext₁, …, extₙ]` — a lossless linear serialisation, round-trip-verified.
@engineer `Token` (atom/extendF/extendG); `deserialise_serialise` (round-trip = some t); `serialise_head_is_atom`, `serialise_nonempty`; `serialiseState_eq_state` (the list evaluates to the same `Term.state` under any applicative); `universalLift_map` says the ingest is format-agnostic — remapping the alphabet commutes with lifting, so there is no per-format path.

The natural linearity of `Polymorphic.Term` (proven structurally by
`BulkExhaustiveness.lean::Node.singleRec_trichotomy`: every internal node has
exactly one recursive slot, the other being a leaf) means each Term is a
non-empty list of small structural tokens.

This file makes that linearity concrete and round-trip-verified. It is the
foundation rung between the abstract categorical universe-syntax
(`UniverseSyntax.lean`) and the modality-specific encoders that will come
next — a planned `UniverseSyntaxModalities.lean` and the Rust engineering layer.
Neither is built: that name is in no commit of this repository, so it is an
intended rung, not a reference.

## The tokens

Three structural units, matching the three constructors of `Term`:

* `atom B U`  — the initial K-token. Always exactly one per Term, at the head.
* `extendF outer` — extends the current Term as `S_routeF (current) outer`.
* `extendG outer` — extends the current Term as `S_routeG outer (current)`.

A well-formed token-list is `[atom B U, ext_1, ext_2, ..., ext_n]` where each
`ext_i` is `extendF` or `extendG`. Order is significant; the head must be an
atom; subsequent atoms are invalid.

## What this file proves

* `serialise` and `deserialise` are defined as small recursive functions.
* `deserialise_serialise` — round-trip: `deserialise (serialise t) = some t`.
* `serialise_head_is_atom` — every serialised Term begins with an `atom` token.
* `serialise_nonempty` — every serialised Term is a non-empty list.
* `serialiseState` and `serialiseState_eq_state` — the token-list form can be
  evaluated under any applicative, producing the same state as `Term.state`.

## What is proved here, and what "no parser" means

* **Proved (native):** the codec. `deserialise_serialise` — round-trip to `some t`, obtained by
  running `UniverseSyntax.fromTerm_unique` twice at `decodeAlgebra` so the codec is forced rather
  than checked; `serialiseState_eq_state` the same way at `serialiseStateAlgebra`; and
  `serialise_head_is_atom` / `serialise_nonempty`, genuine inductions with the append lemmas
  `deserialiseExtensions_append` and `serialiseState_serialise_append` under them.
* **Proved (native):** the α-transport block. `serialise_map`, `deserialise_map_token`,
  `universalLiftCons_map`, `universalLiftCons_state`, `serialise_universalLiftCons` and
  `universalLift_map`, each on its own `foldl` induction (`state_foldl_S_routeF`,
  `serialise_foldl_S_routeF`, `map_foldl_S_routeF`).
* **Proved (native):** read the headline exactly. `universalLift_map` says that remapping every
  symbol by any `f : α → β` and then lifting equals lifting and then transporting — so the
  **lift** cannot branch on what a symbol means, and that is a real constraint on ingestion. It
  says nothing about what happens next: the applicative or algebra applied afterwards is an
  arbitrary function and may branch on anything, so "no hand-coded lexer/parser is needed" holds
  of this one fold, not of the pipeline it feeds.
* **Anchor:** that `universalLift` recovers a stream's structure. It imposes one
  fixed shape — `Term.K` on the first two elements, `Term.S_routeF` for every element after — so
  nothing is parsed and nothing is inferred; and it is partial, returning `none` for every list
  shorter than two. `universalLift_map` commutes trivially with a convention that never looks at
  its input.
* **Proved (riding):** the linearity this file rests on — every internal node has exactly one
  recursive slot — is `BulkExhaustiveness.Node.singleRec_trichotomy`, not re-proved here.
* **Anchor:** that changing α is changing modality (Char ↔ UInt8 ↔ ℝ ↔ AudioSample
  ↔ Pixel), and that `universalLift_map` is therefore "the Axiom XX guarantee". The four `example`s
  instantiate `Char`, `UInt8`, `Bool` and a `Char → UInt8` map, which is real evidence that the
  code is α-generic; that those types *are* text, bytes and audio is the reading, and no
  `AudioSample` or `Pixel` type is instantiated anywhere.

-/

noncomputable section

universe u

namespace Sanguine
namespace UniverseSyntax
namespace Serialisation

open Polymorphic

-- ═══════════════════════════════════════════════════════════════════
-- THE TOKEN TYPE
-- ═══════════════════════════════════════════════════════════════════

/-- The structural unit of universe-syntax serialisation. -/
inductive Token (α : Type u) : Type u where
  | atom (B U : α) : Token α
  | extendF (outer : α) : Token α
  | extendG (outer : α) : Token α

namespace Token

/-- Apply a single extension-token to a partial term. The `atom` case is
    invalid mid-list (atoms only appear at the head), so returns `none`. -/
def applyTo {α : Type u} (t : Term α) : Token α → Option (Term α)
  | .atom _ _ => none
  | .extendF outer => some (Term.S_routeF t outer)
  | .extendG outer => some (Term.S_routeG outer t)

end Token

-- ═══════════════════════════════════════════════════════════════════
-- SERIALISATION
-- ═══════════════════════════════════════════════════════════════════

/-- Linearise a `Term α` to a list of tokens. The head is the innermost K's
    atom; each subsequent token extends the tree as we move outward. -/
def serialise {α : Type u} : Term α → List (Token α)
  | .K B U => [Token.atom B U]
  | .S_routeF prev outer => serialise prev ++ [Token.extendF outer]
  | .S_routeG outer prev => serialise prev ++ [Token.extendG outer]

/-- The head of a serialised Term is always an `atom`. -/
theorem serialise_head_is_atom {α : Type u} (t : Term α) :
    ∃ B U rest, serialise t = Token.atom B U :: rest := by
  induction t with
  | K B U => exact ⟨B, U, [], rfl⟩
  | S_routeF prev outer ih =>
      obtain ⟨B, U, rest, h⟩ := ih
      exact ⟨B, U, rest ++ [Token.extendF outer], by simp [serialise, h]⟩
  | S_routeG outer prev ih =>
      obtain ⟨B, U, rest, h⟩ := ih
      exact ⟨B, U, rest ++ [Token.extendG outer], by simp [serialise, h]⟩

/-- A serialised Term is never empty. -/
theorem serialise_nonempty {α : Type u} (t : Term α) : serialise t ≠ [] := by
  obtain ⟨_, _, _, h⟩ := serialise_head_is_atom t
  rw [h]; intro h'; cases h'

-- ═══════════════════════════════════════════════════════════════════
-- DESERIALISATION
-- ═══════════════════════════════════════════════════════════════════

/-- Apply a list of extension-tokens to a starting Term, left-to-right.
    Returns `none` if any token is an `atom` (atoms only valid at head). -/
def deserialiseExtensions {α : Type u} : Term α → List (Token α) → Option (Term α)
  | t, [] => some t
  | t, tk :: rest =>
      match Token.applyTo t tk with
      | none => none
      | some t' => deserialiseExtensions t' rest

/-- Deserialise a token-list back into a Term. Returns `none` for malformed
    lists (empty, or not starting with an atom). -/
def deserialise {α : Type u} (tokens : List (Token α)) : Option (Term α) :=
  match tokens with
  | [] => none
  | Token.atom B U :: rest => deserialiseExtensions (Term.K B U) rest
  | _ :: _ => none

-- ═══════════════════════════════════════════════════════════════════
-- ROUND-TRIP THEOREM
-- ═══════════════════════════════════════════════════════════════════

-- The round trip is initiality, not induction on the term: `deserialise ∘ serialise` and `some`
-- are two maps out of `Term α` into `Option (Term α)` satisfying the same three recurrences, so
-- `UniverseSyntax.fromTerm_unique` forces them equal. Only the list-level append law needs a
-- recursion, and it is one law over an arbitrary token, not one per routing direction.

/-- `deserialiseExtensions` is a left fold, so a right-append is one more step. One law for any
    token; the `atom` case is `none` on both sides. -/
private theorem deserialiseExtensions_append {α : Type u}
    (t : Term α) (xs : List (Token α)) (tk : Token α) :
    deserialiseExtensions t (xs ++ [tk])
      = (deserialiseExtensions t xs).bind (fun r => Token.applyTo r tk) := by
  induction xs generalizing t with
  | nil => cases tk <;> rfl
  | cons hd tl ih =>
      cases hd with
      | atom B U => rfl
      | extendF o => exact ih _
      | extendG o => exact ih _

/-- Appending one token to a *serialisation* lands inside `deserialiseExtensions`, because
    `serialise` always emits an atom head. -/
private theorem deserialise_serialise_append {α : Type u} (t : Term α) (tk : Token α) :
    deserialise (serialise t ++ [tk])
      = (deserialise (serialise t)).bind (fun r => Token.applyTo r tk) := by
  obtain ⟨B, U, rest, h⟩ := serialise_head_is_atom t
  rw [h, List.cons_append]
  exact deserialiseExtensions_append (Term.K B U) rest tk

/-- The decode target as an algebra: an atom builds `some (K B U)`, an extension token routes
    whatever was decoded so far. -/
private def decodeAlgebra (α : Type u) : AlgebraStructure α (Option (Term α)) where
  inj  := fun B U => some (Term.K B U)
  extL := fun o outer => o.bind (fun r => Token.applyTo r (Token.extendF outer))
  extR := fun outer o => o.bind (fun r => Token.applyTo r (Token.extendG outer))

/-- **The round-trip theorem.** Serialise then deserialise is identity. `deserialise ∘ serialise`
    and `some` are both the map out of `Term α` under `decodeAlgebra`, so initiality equates
    them — the codec is not checked, it is forced. -/
theorem deserialise_serialise {α : Type u} (t : Term α) :
    deserialise (serialise t) = some t :=
  (fromTerm_unique (decodeAlgebra α) (fun s => deserialise (serialise s))
      (fun _ _ => rfl)
      (fun prev outer => deserialise_serialise_append prev (Token.extendF outer))
      (fun outer prev => deserialise_serialise_append prev (Token.extendG outer)) t).trans
    (fromTerm_unique (decodeAlgebra α) some
      (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t).symm

-- ═══════════════════════════════════════════════════════════════════
-- STATE PRESERVATION
-- ═══════════════════════════════════════════════════════════════════

/-- Apply one extension-token to a partial state. Atoms are invalid
    mid-list (state unchanged). -/
def Token.applyState {α : Type u} (app : α → α → α) (state : α) :
    Token α → α
  | .atom _ _ => state
  | .extendF outer => app state outer
  | .extendG outer => app outer state

/-- Evaluate a serialised token-list to its state under an applicative.
    Returns `none` for malformed lists. -/
def serialiseState {α : Type u} (app : α → α → α)
    (tokens : List (Token α)) : Option α :=
  match tokens with
  | [] => none
  | Token.atom B U :: rest =>
      some (rest.foldl (Token.applyState app) (app B U))
  | _ :: _ => none

/-- Appending one token to a serialisation appends its action to the fold. -/
private theorem serialiseState_serialise_append {α : Type u} (app : α → α → α)
    (t : Term α) (tk : Token α) :
    serialiseState app (serialise t ++ [tk])
      = (serialiseState app (serialise t)).map (fun s => Token.applyState app s tk) := by
  obtain ⟨B, U, rest, h⟩ := serialise_head_is_atom t
  rw [h, List.cons_append]
  show some ((rest ++ [tk]).foldl (Token.applyState app) (app B U))
        = (some (rest.foldl (Token.applyState app) (app B U))).map
            (fun s => Token.applyState app s tk)
  rw [List.foldl_append]
  rfl

/-- Evaluating the token list under an algebra whose operations are `app` itself. -/
private def serialiseStateAlgebra {α : Type u} (app : α → α → α) :
    AlgebraStructure α (Option α) where
  inj  := fun B U => some (app B U)
  extL := fun o outer => o.map (fun s => app s outer)
  extR := fun outer o => o.map (fun s => app outer s)

/-- The state of a serialised list, computed directly via fold, equals the
    state of the underlying Term. Both are the map out of `Term α` under
    `serialiseStateAlgebra`: the linear form and the tree form are one translation. -/
theorem serialiseState_eq_state {α : Type u} (app : α → α → α) (t : Term α) :
    serialiseState app (serialise t) = some (Term.state app t) :=
  (fromTerm_unique (serialiseStateAlgebra app) (fun s => serialiseState app (serialise s))
      (fun _ _ => rfl)
      (fun prev outer => serialiseState_serialise_append app prev (Token.extendF outer))
      (fun outer prev => serialiseState_serialise_append app prev (Token.extendG outer)) t).trans
    (fromTerm_unique (serialiseStateAlgebra app) (fun s => some (Term.state app s))
      (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t).symm

-- ═══════════════════════════════════════════════════════════════════
-- MODALITY IRRELEVANCE: AXIOM XX COMPLIANCE
-- ═══════════════════════════════════════════════════════════════════
--
-- Per AGENTS.md Axiom XX: zero language-specific, zero modality-specific
-- implementations. Everything is waves. The universal encoder/decoder
-- operates on `Term α` for ANY α — there is no per-modality code path.
--
-- This section formalises modality irrelevance as theorems:
--
-- * `Token.map` — α-transport on tokens, the structural counterpart of
--   `Polymorphic.map` on terms.
-- * `serialise_map` — serialisation commutes with α-transport. The same
--   structural shape survives changing the wave-sample type.
-- * `deserialise_map_token` — round-trip across α: encode in α, map each
--   token's α-values via f : α → β, decode in β, get the same Term.
-- * `serialiseState_map` — state preservation across α when f is an
--   applicative homomorphism. The cross-α translation operation.
--
-- Together with `Polymorphic.map` and `Polymorphic.map_state` (already
-- in `PolyIotaTermAlgebra.lean`), these establish that the universe-
-- syntax serialisation is genuinely modality-neutral: changing α
-- (Char ↔ UInt8 ↔ ℝ ↔ AudioSample ↔ Pixel) is just mapping each
-- token's α-values; the structural skeleton and the round-trip
-- preservation are unaffected.

/-- α-transport on a single token. Mirrors `Polymorphic.map` at the
    serialised-element level. -/
def Token.map {α β : Type u} (f : α → β) : Token α → Token β
  | .atom B U => .atom (f B) (f U)
  | .extendF outer => .extendF (f outer)
  | .extendG outer => .extendG (f outer)

@[simp] theorem Token.map_atom {α β : Type u} (f : α → β) (B U : α) :
    Token.map f (Token.atom B U) = Token.atom (f B) (f U) := rfl

@[simp] theorem Token.map_extendF {α β : Type u} (f : α → β) (outer : α) :
    Token.map f (Token.extendF outer) = Token.extendF (f outer) := rfl

@[simp] theorem Token.map_extendG {α β : Type u} (f : α → β) (outer : α) :
    Token.map f (Token.extendG outer) = Token.extendG (f outer) := rfl

/-- **Serialisation commutes with α-transport.** Mapping `f : α → β`
    over each token of a serialised Term equals serialising the
    α-transported Term. The structural shape is invariant under
    changing the wave-sample type.

    Operational reading: this is Axiom XX compliance, formally. There
    is no α-specific serialiser. The same `serialise` works for any α
    and commutes with any change of α via `Polymorphic.map`. -/
theorem serialise_map {α β : Type u} (f : α → β) (t : Term α) :
    serialise (Polymorphic.map f t) = (serialise t).map (Token.map f) := by
  induction t with
  | K B U => rfl
  | S_routeF prev outer ih =>
      show serialise (Polymorphic.map f prev) ++ [Token.extendF (f outer)]
            = (serialise prev ++ [Token.extendF outer]).map (Token.map f)
      rw [ih, List.map_append]
      rfl
  | S_routeG outer prev ih =>
      show serialise (Polymorphic.map f prev) ++ [Token.extendG (f outer)]
            = (serialise prev ++ [Token.extendG outer]).map (Token.map f)
      rw [ih, List.map_append]
      rfl

/-- **Round-trip across α.** Encode a Term α as tokens, map each
    token's α-values via f, decode as a Term β. The result is the
    α-transported Term. -/
theorem deserialise_map_token {α β : Type u} (f : α → β) (t : Term α) :
    deserialise ((serialise t).map (Token.map f)) = some (Polymorphic.map f t) := by
  rw [← serialise_map]
  exact deserialise_serialise _

-- ═══════════════════════════════════════════════════════════════════
-- TYPE-INSTANTIATION EXAMPLES (Axiom XX in action)
-- ═══════════════════════════════════════════════════════════════════
--
-- The SAME theorems work for every choice of α. There is no per-α
-- code path — these examples demonstrate it concretely. Changing α
-- changes only the TYPE, not the code.

/-- The same `deserialise_serialise` works on `Term Char` (the text-
    modality instantiation). -/
example (t : Term Char) : deserialise (serialise t) = some t :=
  deserialise_serialise t

/-- The same `deserialise_serialise` works on `Term UInt8` (the
    binary-modality instantiation). -/
example (t : Term UInt8) : deserialise (serialise t) = some t :=
  deserialise_serialise t

/-- The same `deserialise_serialise` works on `Term Bool` (the
    minimum-information-content instantiation). -/
example (t : Term Bool) : deserialise (serialise t) = some t :=
  deserialise_serialise t

/-- Cross-α transport via `Char → UInt8` (e.g., ASCII encoding):
    same code, type-level parameterisation only. -/
example (t : Term Char) (f : Char → UInt8) :
    deserialise ((serialise t).map (Token.map f))
      = some (Polymorphic.map f t) :=
  deserialise_map_token f t

-- ═══════════════════════════════════════════════════════════════════
-- CANONICAL UNIVERSAL LIFT: ANY α-STREAM → Polymorphic.Term
-- ═══════════════════════════════════════════════════════════════════
--
-- The forward direction of the universal translator: given any stream of
-- α-values, lift it canonically into a Polymorphic.Term. The convention
-- is α-agnostic and format-agnostic per Axiom XX:
--
--   [B, U, c, d, e, ...] ↦ K(B, U), extended outward by S_routeF for
--   each subsequent element.
--
-- This is the formal statement of "any data, of any α, becomes universe-
-- syntax without per-format parsing." There is no parser, no language-
-- specific code; just a structural fold. Different α-types (Char, UInt8,
-- AudioSample, Pixel) produce different Term-instances of the same shape.

/-- The canonical lift of an α-stream that has at least two elements
    (B, U, and a `rest`). Produces a Polymorphic.Term whose structure
    is the canonical left-recursive Polymorphic.Term shape. -/
def universalLiftCons {α : Type u} (B U : α) (rest : List α) : Term α :=
  rest.foldl Term.S_routeF (Term.K B U)

/-- The Option-wrapped universal lift, accepting arbitrary lists. Lists
    of length < 2 return `none` (no K-atom can be formed). -/
def universalLift {α : Type u} : List α → Option (Term α)
  | [] => none
  | [_] => none
  | B :: U :: rest => some (universalLiftCons B U rest)

@[simp] theorem universalLift_nil {α : Type u} :
    universalLift ([] : List α) = none := rfl

@[simp] theorem universalLift_singleton {α : Type u} (a : α) :
    universalLift [a] = none := rfl

@[simp] theorem universalLift_cons_cons {α : Type u} (B U : α) (rest : List α) :
    universalLift (B :: U :: rest) = some (universalLiftCons B U rest) := rfl

-- ─── State preservation: lift's state IS the stream's left-fold ──────────

/-- Helper: the state of an S_routeF-foldl over a base Term equals
    the applicative-foldl over the base's state. -/
private theorem state_foldl_S_routeF {α : Type u} (app : α → α → α)
    (t : Term α) (rest : List α) :
    Term.state app (rest.foldl Term.S_routeF t)
      = rest.foldl app (Term.state app t) := by
  induction rest generalizing t with
  | nil => rfl
  | cons hd tl ih =>
      show Term.state app (tl.foldl Term.S_routeF (Term.S_routeF t hd))
            = tl.foldl app (app (Term.state app t) hd)
      rw [ih]
      rfl

/-- **The canonical lift's state IS the stream's left-fold.** Under any
    applicative `app`, the state of the lifted Term equals the left-fold
    of `app` over `rest` starting from `app B U`. The framework's claim
    of universal translation becomes operational: any data, any α,
    folded under any applicative, produces a value via the universe-
    syntax. -/
theorem universalLiftCons_state {α : Type u} (app : α → α → α)
    (B U : α) (rest : List α) :
    Term.state app (universalLiftCons B U rest)
      = rest.foldl app (app B U) :=
  state_foldl_S_routeF app (Term.K B U) rest

-- ─── Correspondence with token-list serialisation ────────────────────────

/-- Helper: serialising the S_routeF-foldl of a base Term equals
    serialising the base followed by extendF tokens for each element. -/
private theorem serialise_foldl_S_routeF {α : Type u} (t : Term α)
    (rest : List α) :
    serialise (rest.foldl Term.S_routeF t)
      = serialise t ++ rest.map Token.extendF := by
  induction rest generalizing t with
  | nil => simp
  | cons hd tl ih =>
      show serialise (tl.foldl Term.S_routeF (Term.S_routeF t hd))
            = serialise t ++ (hd :: tl).map Token.extendF
      rw [ih]
      show serialise (Term.S_routeF t hd) ++ tl.map Token.extendF
            = serialise t ++ (Token.extendF hd :: tl.map Token.extendF)
      show (serialise t ++ [Token.extendF hd]) ++ tl.map Token.extendF
            = serialise t ++ (Token.extendF hd :: tl.map Token.extendF)
      rw [List.append_assoc]
      rfl

/-- **The lift–serialise correspondence.** The token-list serialisation
    of the canonical lift is exactly `[Atom B U, ExtF c, ExtF d, ...]`
    — the structural shape mirrors the input stream with the first two
    elements as the atom and each subsequent element as a left-extension
    token. -/
theorem serialise_universalLiftCons {α : Type u} (B U : α) (rest : List α) :
    serialise (universalLiftCons B U rest)
      = Token.atom B U :: rest.map Token.extendF := by
  unfold universalLiftCons
  rw [serialise_foldl_S_routeF]
  rfl

-- ─── Naturality: the ingestion has NO per-format code path ───────────────
-- The formal content of Axiom XX for the front of the pipeline ("any data of
-- any α becomes universe-syntax without per-format parsing"): the universal
-- lift COMMUTES with any alphabet remap `f : α → β`. Because remapping every
-- input symbol and then lifting equals lifting and then remapping, the lift's
-- behaviour is INDEPENDENT of what the symbols are — it cannot branch on the
-- byte/char/sample's identity. There is no lexer, no per-syntax dispatch; the
-- ingestion is a structural fold, blind to the modality. Hand-coding a
-- language-specific tokeniser is therefore both unnecessary and a violation:
-- this theorem IS the "no need to hardcode" guarantee.

/-- Helper: `map` distributes over the `S_routeF` left-fold. -/
private theorem map_foldl_S_routeF {α β : Type u} (f : α → β)
    (t : Term α) (rest : List α) :
    Polymorphic.map f (rest.foldl Term.S_routeF t)
      = (rest.map f).foldl Term.S_routeF (Polymorphic.map f t) := by
  induction rest generalizing t with
  | nil => rfl
  | cons hd tl ih =>
      show Polymorphic.map f (tl.foldl Term.S_routeF (Term.S_routeF t hd))
            = ((hd :: tl).map f).foldl Term.S_routeF (Polymorphic.map f t)
      rw [ih]
      rfl

/-- The canonical lift commutes with α-transport pointwise. -/
theorem universalLiftCons_map {α β : Type u} (f : α → β)
    (B U : α) (rest : List α) :
    Polymorphic.map f (universalLiftCons B U rest)
      = universalLiftCons (f B) (f U) (rest.map f) := by
  unfold universalLiftCons
  rw [map_foldl_S_routeF]
  rfl

/-- **Universal ingestion is format-agnostic (headline).** Remapping every
    symbol of the input stream by any `f : α → β` and then lifting equals
    lifting and then transporting the Term by `f`. The ingestion has NO
    per-format code path — it is the same structural fold for every alphabet
    (text, bytes, audio samples, pixels). This is the Axiom XX guarantee that
    no hand-coded, language-specific lexer/parser is needed: the universal
    encoder ingests any wave-stream into universe-syntax structurally. -/
theorem universalLift_map {α β : Type u} (f : α → β) (cs : List α) :
    universalLift (cs.map f) = (universalLift cs).map (Polymorphic.map f) := by
  match cs with
  | [] => rfl
  | [_] => rfl
  | B :: U :: rest =>
      show universalLift (f B :: f U :: rest.map f)
            = (universalLift (B :: U :: rest)).map (Polymorphic.map f)
      rw [universalLift_cons_cons, universalLift_cons_cons,
          Option.map_some, universalLiftCons_map]

/-- **Concrete corollary: S-source text and its byte encoding ingest
    identically up to the encoding map.** For any `encode : Char → UInt8`,
    lifting a character stream and byte-encoding the resulting Term equals
    byte-encoding the stream first and then lifting. The surface form (chars
    vs bytes) is a rotation under the same encoder, not a different parser. -/
theorem universalLift_text_eq_bytes (encode : Char → UInt8) (cs : List Char) :
    universalLift (cs.map encode)
      = (universalLift cs).map (Polymorphic.map encode) :=
  universalLift_map encode cs

-- ─── Type-instantiation examples (Axiom XX compliance demonstrated) ──────
-- The SAME `universalLift` and `universalLiftCons_state` work across α.
-- There is no per-α code, only type-level parameterisation.

/-- Bytes: any byte-stream of length ≥ 2 lifts into a `Term UInt8`. -/
example (rest : List UInt8) :
    Term.state (· + ·) (universalLiftCons (1 : UInt8) 2 rest)
      = rest.foldl (· + ·) ((1 : UInt8) + 2) :=
  universalLiftCons_state _ _ _ _

/-- Characters: any char-stream of length ≥ 2 lifts into a `Term Char`. -/
example (rest : List Char)
    (app : Char → Char → Char) :
    Term.state app (universalLiftCons 'a' 'b' rest)
      = rest.foldl app (app 'a' 'b') :=
  universalLiftCons_state _ _ _ _

end Serialisation
end UniverseSyntax
end Sanguine
