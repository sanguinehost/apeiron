import Mathlib.Algebra.Star.Basic
import Mathlib.Algebra.Star.StarAlgHom
import Mathlib.Analysis.CStarAlgebra.Basic
import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Yoneda
import Mathlib.Data.Nat.Digits.Defs
import Substrate.Algebra.Bulk.UniversalEncoding
import Substrate.Seed

/-!
# Universal Translation Theorem (Bulk-Basis Status: Downstream Encoding)

@name Translation Is Just Feeding It In
@child To "translate" something — one language to another, one format to another — you don't need a special converter for each pair. You just hand the thing to a function and let it run. Feeding it in *is* the translation.
@scientist From the Iota basis (one combinator) the identity collapses to `S K K = id` in two lines, and the universal translator becomes application — what took *Principia Mathematica* 362 pages for 1+1=2 reduces to applying a term.
@engineer Part I derives identity (`S K K = id`); Part II proves universal application (∀ f x, (S K K) f x = f x); Parts III–IV give the C*-algebra and Yoneda restatements — translation IS application, in the Iota encoding.

**Bulk-basis note (recently rebased):** under the framework's bulk basis
(`x = B/U` as the generative equation, the bulk as its unbound form), the
Iota / SKI combinator basis is *one valid encoding* of the bulk's
coalgebraic shape — not the foundation itself. The "universal translator"
in the deepest sense IS the bulk's unbound `B/U`; this file proves the
Iota encoding is sufficient as a particular operational realisation.

The Iota-encoding direction proved here:

* Russell needed 362 pages in *Principia Mathematica* to prove `1+1=2`.
* From the Iota basis (one combinator), the Identity collapses to two
  lines (`S K K = id`), and the universal translator becomes application.

This file is load-bearing for the Iota encoding specifically. The
inevitability direction — that the bulk's structure *forces* something
Iota-shaped to exist as an encoding (necessity, not just sufficiency) —
is HALF PROVED, not merely named: `BulkInevitability.contains_K` DERIVES K
in any `BulkAlgebra` (it exhibits `base (fun _ => c) (fun _ => 1)` and closes
on `c / 1 = c`), while S's two orientations and ι are READ BACK — they are
the `BulkAlgebra` structure's own `state_stepB`/`state_stepU` FIELDS, and no
theorem derives that signature from `x = B/U`. `IotaBridge.lean` proves the
sufficiency.

## Structure

1. **Part I — Iota Foundation**: Define S, K, derive Identity, prove `S K K = id`
2. **Part II — Universal Application**: `∀ f x, (S K K) f x = f x`
3. **Part III — C*-Algebraic Translation**: Star-algebra homomorphisms compose
   with identity as the neutral translator.
4. **Part IV — Categorical Universality**: Via Yoneda, morphism evaluation
   is the unique universal map. Translation IS application (in the Iota
   encoding).

## What is proved here, and where the weight actually sits

* **Proved (native):** the cross-α content of Parts V–VI, which is where this file stops being
  `rfl`. `ofBin_toBin` (the `Bool`↔`{0,1}` round-trip via `Nat.digits_lt_base`),
  `translate_preserves_meaning` and `translate_preserves_meaning'` (rotating decimal↔binary leaves
  the `ℕ` floor fixed), `cross_alpha_round_trip` and `cross_alpha_round_trip_canonical`; and on
  the expression side `rpnStack_toRPN` — the stack-fold invariant, a real induction — with
  `ofRPN_toRPN` and `toRPN_preserves_value` following from it.
* **Proved (native):** the Part VII collapse of the decode side onto one function —
  `meaningOf_is_fromTerm`, `meaningOf_congr`, `meaningOf_cons`, `meaningOf_format_agnostic` and
  the `translator_is_universal_encoding` bundle. Each is one `congr 1; funext` over an imported
  fact, so the content is the identification, not a new induction.
* **Proved (Mathlib):** `ofDec_toDec` is `Nat.ofDigits_digits`; the ★-block
  (`star_translation_preserves_add` / `_mul` / `_star`) is `map_add` / `map_mul` / `map_star`
  returned unchanged; `category_translate_id_left` / `_id_right` / `_assoc` are `Category.id_comp`
  / `comp_id` / `assoc` returned unchanged; `yoneda_translation_is_application` is `simp [yoneda]`.
* **Proved (riding):** the whole of Part VIII. `floor_preserves_meaning`,
  `gen_preserves_meaning`, `seed_disk_preserves_meaning` and the `section_is_the_seed` bundle are
  each one `rw` of `Seed.rexpand_rcollapse`, `Seed.Gen.decode_encode` or
  `Seed.loadBytes_saveBytes`. "Supply the prediction, correctness is free" is those three
  theorems, cited.
* **Proved (riding):** read the format-agnosticity claim exactly.
  `Serialisation.universalLift_map` — restated here as `meaningOf_format_agnostic` — says
  ingestion commutes with any alphabet remap `f : α → β`, so the lift cannot branch on what a
  symbol means. It says nothing about the applicative applied afterwards, which is an arbitrary
  `α → α → α` and may branch on anything. Likewise `UniverseSyntax.fromTerm_unique` is a
  uniqueness theorem about a structure-respecting map out of the Term algebra; it is not a
  theorem about semantics, and composing it into `meaningOf_is_fromTerm` does not make it one.
* **Anchor:** Parts I–II, as claims about translation. `SKK_applied_eq`,
  `SKK_eq_I`, `SKK_eq_id`, `universal_application`, `translate_is_apply`, `translation_composes`
  and `universal_translation` are all `rfl` on `I_comb x = x`, which is `I_comb`'s own defining
  equation; `T` is called a translator only in its binder name, and the docstring on
  `universal_translation` already says so. That Russell's 362 pages "reduce to `rfl`" compares two
  different statements — nothing here proves `1 + 1 = 2`, or relates the Iota basis to
  *Principia*'s development.
* **Anchor:** that the `ℕ` value is "the meaning floor", that `Expr.eval` is "the
  meaning", and that the base is "the applicative, the only freedom". The theorems are about digit
  lists, expression trees and folds; that any of it is *meaning* is the reading. Its evidence
  outside the Lean is the `.seed` round-trip and the model rows measured in
  `papers/paper1/src/11-limits.md` (Intel Core i9-14900KF / RTX 5080, agreement checked before
  time).

-/

noncomputable section

universe u v

namespace Sanguine

-- ═══════════════════════════════════════════════════════════════════════════
-- PART I: THE IOTA FOUNDATION
-- ═══════════════════════════════════════════════════════════════════════════

variable {α β γ : Type*}

/-- The S combinator: `S f g x = f x (g x)`.
    Distributes application across a pair of functions. -/
def S_comb (f : α → β → γ) (g : α → β) (x : α) : γ := f x (g x)

/-- The K combinator: `K x y = x`.
    The constant function — selects the first argument. -/
def K_comb (x : α) (_ : β) : α := x

/-- The I combinator (Identity): `I x = x`.
    Derived, not primitive. -/
def I_comb (x : α) : α := x

/-- **The Two-Line Proof.**
    `S K K x = x` for all `x`. The identity function is not primitive — it is
    derived from the applicator (S) and the selector (K). Russell's 362 pages
    reduce to `rfl` when you have the right basis.

    Proof: `S K K x = K x (K x) = x`. -/
theorem SKK_applied_eq (x : α) :
    S_comb (K_comb (β := α → α)) (K_comb (β := α)) x = x := by rfl

/-- `S K K = I` as functions. -/
theorem SKK_eq_I :
    (fun x : α => S_comb (K_comb (β := α → α)) (K_comb (β := α)) x) = @I_comb α := by
  funext x; rfl

/-- `S K K = id`. -/
theorem SKK_eq_id :
    (fun x : α => S_comb (K_comb (β := α → α)) (K_comb (β := α)) x) = id := by
  funext x; rfl

-- ═══════════════════════════════════════════════════════════════════════════
-- PART II: UNIVERSAL APPLICATION
-- ═══════════════════════════════════════════════════════════════════════════

/-- **Universal Application.**
    For any function `f` and any input `x`, the identity combinator
    S(K)(K) applied to `f` and `x` yields `f(x)`.

    This IS the Universal Translator: it doesn't care what `f` is.
    `f` could translate English→Mandarin, S→x86, Voice→Code.
    The translator is a *parameter*, not hardcoded infrastructure. -/
theorem universal_application (f : α → β) (x : α) :
    I_comb f x = f x := by rfl

/-- The Universal Translator is the evaluation map: given a translator `T`
    and an input `x`, it returns `T x`. This is function application. -/
theorem translate_is_apply (T : α → β) (x : α) :
    I_comb T x = T x := by rfl

/-- Translation composes: if `T₁ : A → B` and `T₂ : B → C`, then
    `T₂ ∘ T₁ : A → C` is also a translation. -/
theorem translation_composes (T₁ : α → β) (T₂ : β → γ) (x : α) :
    I_comb (T₂ ∘ T₁) x = T₂ (T₁ x) := by rfl

-- ═══════════════════════════════════════════════════════════════════════════
-- PART III: C*-ALGEBRAIC TRANSLATION
-- ═══════════════════════════════════════════════════════════════════════════

section CStarTranslation

set_option linter.unusedSectionVars false

variable {R : Type*} [CommSemiring R] [StarRing R]
variable {A : Type*} [Semiring A] [Algebra R A] [StarRing A] [StarModule R A]
variable {B : Type*} [Semiring B] [Algebra R B] [StarRing B] [StarModule R B]
variable {Γ : Type*} [Semiring Γ] [Algebra R Γ] [StarRing Γ] [StarModule R Γ]

/-- **Identity Translation.**
    The identity ★-homomorphism maps every element to itself.
    C*-algebraic form of `S K K = id`. -/
theorem star_identity_translation (a : A) :
    StarAlgHom.id R A a = a := by rfl

/-- **Translation preserves addition.** -/
theorem star_translation_preserves_add (φ : A →⋆ₐ[R] B) (a₁ a₂ : A) :
    φ (a₁ + a₂) = φ a₁ + φ a₂ := map_add φ a₁ a₂

/-- **Translation preserves multiplication.** -/
theorem star_translation_preserves_mul (φ : A →⋆ₐ[R] B) (a₁ a₂ : A) :
    φ (a₁ * a₂) = φ a₁ * φ a₂ := map_mul φ a₁ a₂

/-- **Translation preserves star (involution).**
    "Meaning" (the star structure) is invariant under translation. -/
theorem star_translation_preserves_star (φ : A →⋆ₐ[R] B) (a : A) :
    φ (star a) = star (φ a) := map_star φ a

/-- **Composition of translations is a translation.**
    English→Mandarin composed with Mandarin→x86 gives English→x86. -/
theorem star_translation_composition (φ : A →⋆ₐ[R] B) (ψ : B →⋆ₐ[R] Γ) (a : A) :
    (ψ.comp φ) a = ψ (φ a) := by rfl

/-- **Left identity law.** Identity translator composes transparently. -/
theorem star_translate_id_left (φ : A →⋆ₐ[R] B) :
    φ.comp (StarAlgHom.id R A) = φ := by ext a; rfl

/-- **Right identity law.** -/
theorem star_translate_id_right (φ : A →⋆ₐ[R] B) :
    (StarAlgHom.id R B).comp φ = φ := by ext a; rfl

/-- **Associativity of translation.** -/
theorem star_translate_assoc
    {D : Type*} [Semiring D] [Algebra R D] [StarRing D] [StarModule R D]
    (φ : A →⋆ₐ[R] B) (ψ : B →⋆ₐ[R] Γ) (χ : Γ →⋆ₐ[R] D) :
    χ.comp (ψ.comp φ) = (χ.comp ψ).comp φ := by ext a; rfl

end CStarTranslation

-- ═══════════════════════════════════════════════════════════════════════════
-- PART IV: CATEGORICAL UNIVERSALITY (via Yoneda)
-- ═══════════════════════════════════════════════════════════════════════════

section CategoricalTranslation

open CategoryTheory

/-- **The Yoneda Evaluation is the Universal Translator.**

    `yoneda.map f` applied to `id_X` recovers `f`.
    The bijection `Nat(Hom(−,A), Hom(−,B)) ≅ Hom(A,B)` is given by
    evaluation at the identity — which is `S(K)(K)` applied to the scheme.

    **Translation IS application.** -/
theorem yoneda_translation_is_application
    {C : Type u} [Category.{v} C] (X Y : C) (f : X ⟶ Y) :
    (yoneda.map f).app (Opposite.op X) (𝟙 X) = f := by
  simp [yoneda]

/-- **Left identity in any category.** -/
theorem category_translate_id_left
    {C : Type u} [Category.{v} C] {X Y : C} (f : X ⟶ Y) :
    𝟙 X ≫ f = f :=
  Category.id_comp f

/-- **Right identity in any category.** -/
theorem category_translate_id_right
    {C : Type u} [Category.{v} C] {X Y : C} (f : X ⟶ Y) :
    f ≫ 𝟙 Y = f :=
  Category.comp_id f

/-- **Associativity of translation in any category.** -/
theorem category_translate_assoc
    {C : Type u} [Category.{v} C] {W X Y Z : C}
    (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z) :
    (f ≫ g) ≫ h = f ≫ (g ≫ h) :=
  Category.assoc f g h

end CategoricalTranslation

-- ═══════════════════════════════════════════════════════════════════════════
-- THE COMPLETE THEOREM
-- ═══════════════════════════════════════════════════════════════════════════

/-- **The Universal Translation Theorem.**

    For any function `T : α → β` (a translator between ANY two domains)
    and any input `x : α`:

    1. `S(K)(K)` is the identity (`SKK_eq_id`)
    2. Applying the identity to T and x yields T(x) (`universal_application`)
    3. T(x) preserves structure when T is a ★-homomorphism
    4. The Yoneda lemma guarantees this is the UNIQUE universal construction

    Honest scope of *this* theorem: `universal_translation` itself is
    `I_comb T x = T x`, and since `I_comb = id` this is **`rfl`** — the
    identity-combinator triviality, saying nothing about translation beyond
    "`id` applied is `id`." The substantive content lives in the parts it
    bundles: the ★-homomorphism preservation (3) and the Yoneda uniqueness
    (4) are genuine Mathlib-level results; the "translation IS application"
    reading rests on *those*, plus the file-header **sufficiency** claim (the
    Iota encoding is *one* valid encoding; inevitability is half proved —
    `BulkInevitability.contains_K` derives K in any `BulkAlgebra`, while S and ι
    are the signature's own `state_stepB`/`state_stepU` fields read back). So:
    in the Iota encoding the neutral translator is `id` —

        output = T(input)

    — and the weight is on (3)–(4), not on this one-line identity. -/
theorem universal_translation {α β : Type*} (T : α → β) (input : α) :
    I_comb T input = T input := by rfl

-- ═══════════════════════════════════════════════════════════════════════════
-- PART V: THE CROSS-α TRANSLATOR (decimal ↔ binary, meaning preserved)
-- ═══════════════════════════════════════════════════════════════════════════

/-!
Parts I–IV say *translation is application* but the headline theorems are `rfl`
(`id` applied is `id`). This part gives that statement real content on a genuinely
cross-α pair, the smallest non-trivial formal atom: **decimal `List ℕ` ↔ binary
`List Bool`**, two genuinely different surface alphabets.

The pipeline is exactly `translate(α → β) = express_β ∘ collapse-to-meaning-floor ∘
lift_α`:

* `lift_α` / `collapse-to-meaning-floor` = `decode` = `Nat.ofDigits b`, the Horner
  left-fold. This IS application: `Nat.ofDigits b` is `Term.state` under the base-`b`
  applicative `acc ↦ d ↦ acc + b·…` (`universal_application`; `OneLift`'s
  `lift_and_recover_are_one_apply` is the general "state = fold").
* the **meaning floor** is the `ℕ` value — invariant under the surface, the content
  Parts I–IV only gestured at.
* `express_β` = `encode` = `Nat.digits b`.

`translate_preserves_meaning` is the heart (NOT `rfl`): rotating the surface from
decimal to binary leaves the floor fixed. `cross_alpha_round_trip` closes the G1 gap
— lift at α_source, express at a *different* α_target, and back is identity on the
meaning — the cross-α analogue of `Navigation.round_trip_id` (which was same-α).

## The three faces, and the files we operate from

The translator is read on the trinity (`HashTrinity.lean`, `TrinityDepth.lean`):

* **B/U (the fold / the merge) — `decode`.** `ofDec`/`ofBin` = `Nat.ofDigits b` is the
  **B/U accumulator** of `HashTrinity` (`merge_BU`, the monoid combine), the Horner
  left-fold over the digit suffix. The **base `b` is the merge's only freedom** —
  decimal `b=10`, binary `b=2` — and that is the *entire* difference between the two
  alphabets (`merge_is_the_only_freedom`). `/`-total: the fold reaches the unique floor
  (`TrinityDepth.defeq_is_the_trinity_at_depth`, the Being-cell face).
* **B (the monad / the value) — the meaning floor.** The `ℕ` is the unit each surface
  shares (`HashTrinity` Map face; `TrinityDepth` monad, the reflexive no-op `a ≡ a`).
  The round-trip returns here; `translate_preserves_meaning` is its invariance.
* **NB (the Null / erasure) — the surface alphabet (the α).** Decimal-vs-binary is the
  representation redundancy that carries no meaning and **cancels** under translation
  (`TrinityDepth` NB face: equivalent forms cancel, `null_times_mirror`). The surface IS
  the Non-Being relative to the value.

**Depth 2, never deeper** (`TrinityDepth`, the depth measure; THE_HASHMAP_PATH:
"everything is a hashmap or a recursor with no more than a depth of 2"). `lift_α` is the
**depth-1 decoupled** Horner fold over the digit suffix (the streaming law, each digit
read into the accumulator from its own coordinate). The cross-α step is **one coupling**
through the shared meaning floor (`express_β` reads the same `ℕ`): depth 2. The whole
translator is the B/U accumulator (the hashmap) = the fold (the recursor) at depth 2. -/
section CrossAlphaTranslator

/-- Decimal surface: base-10 digits (each `< 10`). `decode = Nat.ofDigits 10`. -/
def ofDec (ds : List ℕ) : ℕ := Nat.ofDigits 10 ds
/-- `express` to the decimal surface — the canonical base-10 digit list. -/
def toDec (n : ℕ) : List ℕ := Nat.digits 10 n

/-- Binary surface: a genuinely different alphabet, `List Bool`. `decode` reads each
    bit as `0`/`1` and Horner-folds base 2. -/
def ofBin (bs : List Bool) : ℕ := Nat.ofDigits 2 (bs.map (fun b => if b then 1 else 0))
/-- `express` to the binary surface — the canonical bit list. -/
def toBin (n : ℕ) : List Bool := (Nat.digits 2 n).map (fun d => decide (d = 1))

/-- Decimal codec inverts (the canonical section): straight from `Nat.ofDigits_digits`. -/
theorem ofDec_toDec (n : ℕ) : ofDec (toDec n) = n := Nat.ofDigits_digits 10 n

/-- Binary codec inverts. The only content beyond `Nat.ofDigits_digits` is that the
    `Bool`↔`{0,1}` round-trip is the identity on real binary digits (each `< 2`). -/
theorem ofBin_toBin (n : ℕ) : ofBin (toBin n) = n := by
  unfold ofBin toBin
  rw [List.map_map]
  have hid : ∀ d ∈ Nat.digits 2 n,
      ((fun b => if b then 1 else 0) ∘ (fun d => decide (d = 1))) d = id d := by
    intro d hd
    have hlt : d < 2 := Nat.digits_lt_base (by norm_num) hd
    have : d = 0 ∨ d = 1 := by omega
    rcases this with h | h <;> subst h <;> rfl
  rw [List.map_congr_left hid, List.map_id]
  exact Nat.ofDigits_digits 2 n

/-- The cross-α translator: lift the decimal surface to the meaning floor, express it
    on the binary surface. `express_bin ∘ decode_dec`. -/
def translate_dec_bin (ds : List ℕ) : List Bool := toBin (ofDec ds)
/-- The reverse rotation. -/
def translate_bin_dec (bs : List Bool) : List ℕ := toDec (ofBin bs)

/-- **The heart — meaning is invariant under the alphabet rotation (NOT `rfl`).**
    Translating decimal→binary changes the surface (a `List Bool`, a different
    alphabet) but the `ofBin` reading of the output equals the `ofDec` reading of the
    input: the meaning floor is preserved. This is the real content of "translation
    IS application" the `rfl` headlines only gestured at. -/
theorem translate_preserves_meaning (ds : List ℕ) :
    ofBin (translate_dec_bin ds) = ofDec ds := by
  unfold translate_dec_bin; rw [ofBin_toBin]

/-- The reverse: binary→decimal preserves the floor. -/
theorem translate_preserves_meaning' (bs : List Bool) :
    ofDec (translate_bin_dec bs) = ofBin bs := by
  unfold translate_bin_dec; rw [ofDec_toDec]

/-- **G1 closed — the cross-α round-trip.** Lift at α_source (decimal), express at a
    genuinely different α_target (binary), and back: identity on the meaning. The
    cross-α analogue of `Navigation.round_trip_id` (which was same-applicative). -/
theorem cross_alpha_round_trip (ds : List ℕ) :
    ofDec (translate_bin_dec (translate_dec_bin ds)) = ofDec ds := by
  rw [translate_preserves_meaning', translate_preserves_meaning]

/-- On the canonical decimal form, the round-trip is bit-identical on the surface
    itself (not just on the meaning) — the strongest form, the runnable witness. -/
theorem cross_alpha_round_trip_canonical (n : ℕ) :
    translate_bin_dec (translate_dec_bin (toDec n)) = toDec n := by
  unfold translate_bin_dec translate_dec_bin
  rw [ofDec_toDec, ofBin_toBin]

end CrossAlphaTranslator

-- ═══════════════════════════════════════════════════════════════════════════
-- PART VI: MEANING IS A COMPUTATION — the floor is the VALUE (translation IS application)
-- ═══════════════════════════════════════════════════════════════════════════

/-!
Part V's meaning floor was a `ℕ` recovered by a codec. Here the floor is a genuine
**computation** — the *value* of an arithmetic expression — and translation between two
notations (a structured tree surface ↔ a flat RPN token surface) preserves it. The meaning
IS `eval`, the fold over the expression with the `{+, ×}` algebra: the recursor path
(`THE_HASHMAP_PATH` §the dual), the eliminator `I`, the algebra the only freedom — which is
exactly **"translation IS application"** (`universal_application`) made concrete on a real
computation, not a representation. Losslessness ⟹ value-preservation, the same rule as
`SangAST.lossless_preserves_meaning` — and that rule is exactly `congrArg m h`, one rewrite under CITATION-EXEMPT: rides the rewrite only; the same sentence disclaims that docstring's larger conclusions
`recover ∘ lift = id`, true of any function whatever. Take it for that and no more: the larger
conclusions its docstring draws there — that the syntactic→semantic "gap" dissolves, that
undecidable program equivalence never applied — are recorded as unproved in that module's own
warrant block, and nothing below leans on them. -/
section ComputationTranslator

/-- A tiny arithmetic expression — the structured surface (α = tree). -/
inductive Expr where
  | lit : ℕ → Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
deriving Repr, DecidableEq

namespace Expr

/-- **The meaning floor — the VALUE.** `eval` is the fold over the tree (the recursor path);
    the value it computes is the meaning, invariant under the surface. eval IS application. -/
def eval : Expr → ℕ
  | lit n => n
  | add l r => l.eval + r.eval
  | mul l r => l.eval * r.eval

/-- An RPN token — the flat (postfix) surface alphabet, genuinely different from the tree. -/
inductive Tok where
  | num : ℕ → Tok
  | plus
  | times
deriving Repr, DecidableEq

/-- `express` to the RPN surface — postfix serialization (a fold). -/
def toRPN : Expr → List Tok
  | lit n => [Tok.num n]
  | add l r => l.toRPN ++ r.toRPN ++ [Tok.plus]
  | mul l r => l.toRPN ++ r.toRPN ++ [Tok.times]

/-- `lift` the RPN surface — the **stack fold** (the recursor path, NO precedence parser):
    push a number, combine the top two on an operator. Total (malformed input skips). -/
def rpnStack : List Tok → List Expr → List Expr
  | [], stack => stack
  | Tok.num n :: rest, stack => rpnStack rest (lit n :: stack)
  | Tok.plus :: rest, b :: a :: stack => rpnStack rest (add a b :: stack)
  | Tok.times :: rest, b :: a :: stack => rpnStack rest (mul a b :: stack)
  | _ :: rest, stack => rpnStack rest stack

/-- Deserialize RPN to an expression (the lone term on the final stack). -/
def ofRPN (toks : List Tok) : Option Expr :=
  match rpnStack toks [] with
  | [e] => some e
  | _ => none

/-- **Folding an expression's RPN onto the stack pushes that expression** — the key stack-fold
    invariant, the recursor-path identity that makes RPN a faithful surface. -/
theorem rpnStack_toRPN (e : Expr) :
    ∀ rest stack, rpnStack (toRPN e ++ rest) stack = rpnStack rest (e :: stack) := by
  induction e with
  | lit n => intro rest stack; rfl
  | add l r ihl ihr =>
      intro rest stack; simp only [toRPN, List.append_assoc]; rw [ihl, ihr]; rfl
  | mul l r ihl ihr =>
      intro rest stack; simp only [toRPN, List.append_assoc]; rw [ihl, ihr]; rfl

/-- **The RPN surface round-trips losslessly** — `ofRPN ∘ toRPN = some`. -/
theorem ofRPN_toRPN (e : Expr) : ofRPN (toRPN e) = some e := by
  unfold ofRPN
  rw [show toRPN e = toRPN e ++ [] from (List.append_nil _).symm, rpnStack_toRPN]
  rfl

/-- **The value is preserved across the notation change (the heart).** Translating an
    expression to RPN and back recovers it exactly, so its `eval` — the meaning floor, a real
    computation — is preserved. The meaning is the value; the notation is the surface; the value
    is invariant. "Translation IS application," on a computation. -/
theorem toRPN_preserves_value (e : Expr) :
    (ofRPN (toRPN e)).map eval = some e.eval := by
  rw [ofRPN_toRPN, Option.map_some]

end Expr
end ComputationTranslator

-- ═══════════════════════════════════════════════════════════════════════════
-- PART VII: THE UNIFICATION — the translator IS `universalLift` / `state` / `fromTerm`
-- ═══════════════════════════════════════════════════════════════════════════

/-!
Parts V/VI/§5 built concrete translators with **bespoke** decoders (`ofDec`, `Expr.eval`,
`ofToks`). That is parallel machinery — the very per-format codec `UniversalEncoding` proves you
never need. This part collapses the **decode** side into the ONE universal function: the meaning
floor of any surface is `Term.state app ∘ universalLift` = `fromTerm (stateAlgebra app) ∘
universalLift` — the **unique** map out (`UniversalEncoding` fact 4, `state_eq_fromTerm_stateAlgebra`),
**applicative-parametric** (fact 3, `applicative_is_the_only_freedom`), **format-agnostic in**
(fact 5, `universalLift_map`). The base/applicative is the only freedom; the surface is an α.
Decimal and binary are this ONE decoder at two applicatives. -/
section TranslatorUnified

open Polymorphic UniverseSyntax UniverseSyntax.Serialisation

/-- **The ONE universal decoder.** Lift any surface (format-agnostic), read its meaning as the
    `state` under the chosen applicative. No bespoke codec: one function, parameterized by α (the
    surface) and the applicative (the meaning). -/
def meaningOf {α : Type} (app : α → α → α) (cs : List α) : Option α :=
  (universalLift cs).map (Term.state app)

/-- The universal decoder IS the Horner left-fold (`universalLiftCons_state`): the meaning is the
    accumulator threaded by the applicative — for ANY applicative. -/
theorem meaningOf_cons {α : Type} (app : α → α → α) (B U : α) (rest : List α) :
    meaningOf app (B :: U :: rest) = some (rest.foldl app (app B U)) := by
  simp only [meaningOf, universalLift_cons_cons, Option.map_some, universalLiftCons_state]

/-- **The decoder is `fromTerm` at the state algebra — the UNIQUE map out (fact 4).** The meaning
    floor is forced, not a codec choice; there is no room for a bespoke decoder. -/
theorem meaningOf_is_fromTerm {α : Type} (app : α → α → α) (cs : List α) :
    meaningOf app cs = (universalLift cs).map (fromTerm (stateAlgebra α app)) := by
  simp only [meaningOf]
  congr 1; funext t; exact state_eq_fromTerm_stateAlgebra app t

/-- **Meaning is the applicative (fact 3, the congruence).** Two applicatives agreeing pointwise
    give the same meaning for every surface — the meaning depends only on the applicative, never on
    the surface presentation. "The base is the only freedom," made literal. -/
theorem meaningOf_congr {α : Type} {app₁ app₂ : α → α → α} (h : ∀ x y, app₁ x y = app₂ x y)
    (cs : List α) : meaningOf app₁ cs = meaningOf app₂ cs := by
  simp only [meaningOf]
  congr 1; funext t; exact applicative_is_the_only_freedom h t

/-- **Format-agnostic in (fact 5, `universalLift_map`).** Relabel the surface alphabet by any
    `f : α → β`; lifting commutes, so the decoder cannot branch on what a symbol means — there is
    no per-format code path. -/
theorem meaningOf_format_agnostic {α β : Type} (app : β → β → β) (f : α → β) (cs : List α) :
    (universalLift (cs.map f)).map (Term.state app)
      = ((universalLift cs).map (Polymorphic.map f)).map (Term.state app) := by
  rw [universalLift_map]

/-- A base-`b` numeral applicative — the Horner step `acc ↦ d ↦ acc·b + d`. The decimal and binary
    decoders are `meaningOf (hornerApp 10)` and `meaningOf (hornerApp 2)`: ONE `meaningOf`, the base
    the only freedom. -/
def hornerApp (b : ℕ) : ℕ → ℕ → ℕ := fun acc d => acc * b + d

-- The ONE universal decoder reads decimal `[4,2]` and binary `[1,0,1,0,1,0]` to the SAME `42`,
-- differing only in the applicative (the base) — the literal "meaning is the applicative."
example : meaningOf (hornerApp 10) [4, 2] = some 42 := by rw [meaningOf_cons]; rfl
example : meaningOf (hornerApp 2) [1, 0, 1, 0, 1, 0] = some 42 := by rw [meaningOf_cons]; rfl

/-- **The unification (headline): the translator's decode IS `UniversalEncoding`'s one function.**
    The meaning floor of ANY surface is `meaningOf app`, which is

    1. `fromTerm` at the state algebra (`meaningOf_is_fromTerm`) — the unique map out (fact 4);
    2. determined by the applicative alone (`meaningOf_congr`) — fact 3, the base the only freedom;
    3. format-agnostic in (`meaningOf_format_agnostic`) — fact 5, no per-format branch.

    So the **decode mechanism** is no longer bespoke: it is this one universal function, and
    decimal `meaningOf (hornerApp 10)` / binary `meaningOf (hornerApp 2)` are it at two
    applicatives — the base the only freedom. The concrete decoders of Parts V/VI/§5 are instances
    of this same pattern (a fold = `state` at the meaning applicative); literally rewiring each onto
    `meaningOf` is a labeled refactor (decimal differs only by digit order from `ofDec`; `eval` and
    `normalize` factor through `Term`'s initiality). The per-codec *section* (the encode/decode
    round-trip inverse — `ofDigits_digits`, the stack folds) is the one genuinely instance-specific
    part; the meaning extraction is the single `fromTerm`. -/
theorem translator_is_universal_encoding {α : Type} (app : α → α → α) (cs : List α) :
    (meaningOf app cs = (universalLift cs).map (fromTerm (stateAlgebra α app)))
      ∧ (∀ app₂ : α → α → α, (∀ x y, app x y = app₂ x y) → meaningOf app cs = meaningOf app₂ cs) :=
  ⟨meaningOf_is_fromTerm app cs, fun _ h => meaningOf_congr h cs⟩

end TranslatorUnified

-- ═══════════════════════════════════════════════════════════════════════════
-- PART VIII: THE SECTION IS THE SEED — the round-trip is free, not hand-proved
-- ═══════════════════════════════════════════════════════════════════════════

/-!
Part VII collapsed the **decode** (meaning extraction) into `fromTerm`/`state`. This collapses the
**section** (the encode/decode round-trip) into `Substrate/Seed.lean`. A `Seed.Floor` (Being `key`
+ Null `delta` residue + `recover`) or a `Seed.Gen` (`predict` + `step`) is a lossless codec whose
round-trip is a theorem proven ONCE for ANY key/predictor (`rexpand_rcollapse`,
`genDecode_genEncode`) — *"supply the prediction, correctness is free."* Composed with the generic
`lossless ⟹ meaning-preserved` law, the section preserves ANY meaning with no per-codec proof.

So the bespoke serializers of Parts V/VI/§5 (and their hand-proved inverses `ofRPN_toRPN`,
`ofToks_toToks`, `ofBin_toBin`) were the per-format codec this subsumes: the round-trip is not
instance-specific, it is `Seed`. (Honest scope: those serializers are *structural* — tree ↔ tokens,
positional numeral — not literally the per-element `Floor`/`Gen` word-stream codec, so re-expressing
each as a `Floor`/`Gen` is a presentation choice; what is fully generic, and proven here, is that the
section's *round-trip and meaning-preservation are free for any Seed codec*. The decode itself is the
boundary readout `NullReadout.decodeNull` — an O(1) navigation, not a parse.) -/
section SectionIsTheSeed

open Sanguine.Seed

/-- **A `Seed.Floor` codec preserves ANY meaning, for free.** The round-trip is `rexpand_rcollapse`
    (proven once for every Floor); meaning-preservation is then immediate, no per-codec proof. -/
theorem floor_preserves_meaning {M : Type} (F : Floor) (m : Tensor → M) (t : Tensor) :
    m (rexpand F (rcollapse F t)) = m t := by rw [rexpand_rcollapse]

/-- **A `Seed.Gen` (any predictor) codec preserves ANY meaning, for free.** Round-trip is
    `genDecode_genEncode`; the prediction only changes the residual *size*, never the meaning. -/
theorem gen_preserves_meaning {Ctx M : Type} (g : Gen Ctx) (m : Tensor → M) (t : Tensor) :
    m (g.decode (g.encode t)) = m t := by rw [Gen.decode_encode]

/-- **The disk codec preserves ANY meaning, for free.** Save to the `.seed` byte stream and load it
    back (`loadBytes_saveBytes`); the meaning survives the whole collapse + varint pipeline. -/
theorem seed_disk_preserves_meaning {M : Type} (F : Floor) (m : Tensor → M) (t : Tensor) :
    (loadBytes F (saveBytes F t).length (saveBytes F t)).map m = some (m t) := by
  rw [loadBytes_saveBytes, Option.map_some]

/-- **The section is the Seed (headline).** Any lossless codec the translator needs is a
    `Seed.Floor` or `Seed.Gen`: the round-trip is a theorem proven ONCE for every key/predictor
    (`rexpand_rcollapse` / `genDecode_genEncode`), and meaning-preservation follows for free
    (`floor_preserves_meaning` / `gen_preserves_meaning`). There is no bespoke per-format codec and
    no per-codec inverse proof — *"supply the prediction, correctness is free."* -/
theorem section_is_the_seed {Ctx M : Type} (F : Floor) (g : Gen Ctx) (m : Tensor → M) (t : Tensor) :
    (rexpand F (rcollapse F t) = t)
      ∧ (g.decode (g.encode t) = t)
      ∧ (m (rexpand F (rcollapse F t)) = m t)
      ∧ (m (g.decode (g.encode t)) = m t) :=
  ⟨rexpand_rcollapse F t, Gen.decode_encode g t,
   floor_preserves_meaning F m t, gen_preserves_meaning g m t⟩

end SectionIsTheSeed

end Sanguine
