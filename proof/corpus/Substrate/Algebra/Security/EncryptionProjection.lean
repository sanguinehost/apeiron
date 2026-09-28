import Substrate.StationaryPhase.BiasInjectedRecovery
import Mathlib.Logic.Function.Basic
import Mathlib.Data.Set.Card

/-!
# What Encryption Is — the Being-Cell Bijection vs the Null-Cell Projection
@name Lock vs Shredder
@child There are two kinds of scrambling: a lock you can always undo with the right key, and a shredder that genuinely throws information away. Which one you have decides whether you can ever get the original back.
@scientist Encryption is one universal encoding read at an applicative; the only question is whether the map is injective (a cipher, the Being cell, `/` total) or collapses (a hash, the Null cell, `/` singular), and that fork is the NullCell fork.
@engineer `decryption_is_unique_floor` shows inverting an injective cipher is the one collapse to a unique floor (no search); `collision_breaks_unique_floor` shows a collision leaves only a fiber, and `cipher_fiber_is_a_singleton` / `collision_fiber_is_not_a_singleton` are the security fork as a cardinal — a bijection's ciphertext admits exactly one plaintext, a collision's admits no single one.

**This fork is universal — it is `collapse ⊣ select`.** The injective/Being side is
*recoverable*: its inverse is the `select`/decode face, the one collapse to a unique floor
(`decryption_is_unique_floor`), `O(log n)` not `2^n`, never a search. The collapsing/Null side
is the one-way wall (`collision_breaks_unique_floor`). The same cut recurs as instances across
the corpus: the Born phase the collapsing Touch discards is exactly the sparse fiber `select`
recovers (`PhaseBlindSelect`); the in-frame descent is the cipher whose decryptor is the descent,
off-frame the wall (`DescentIsTheEasyInverse`); the Null web's recovering face (`NullWeb`); the
runtime no-op-vs-reduce step (`EvaluatorReduceStep`). Recovery is the unit-licensed inverse face
of collapse — a unit (injectivity / in-frame / `IsUnit(pairing)`) licenses the select; off-frame,
no unit, the wall.

There is no primitive encryption/decryption — there is one `UniversalEncoding`, read at an
applicative, and the only structural question is whether that map is injective (the Being
cell) or collapses (the Null cell). That single fork is the `NullCell` fork:

| | **cipher** (a bijection) | **hash** (a collapse) |
|---|---|---|
| cell | Being — `/` total (`NullCell.phase_invertible`) | Null — `/` singular (`null_not_invertible`) |
| collision | none (`phase_no_zero_divisors`) | `(1+j)(1−j)=0` (`null_times_mirror`) |
| Shannon entropy | preserved | reduced (many-to-one) |
| inverse | unique (`B/(B/U)=U`) | a class (a fiber), not a point |

## The two theorems

* **Inverting any injective cipher IS the one collapse** (`decryptRecovery`,
  `decryption_is_unique_floor`). The fiber-indicator residual aimed at a ciphertext has a
  unique floor at the plaintext, *because* the map is injective — a `BiasInjectedRecovery`
  with a unique global minimum, no search. A bijection destroyed nothing, so recovery is
  `B/(B/U)=U`, the only cost evaluating the applicative's `/`. The classical hardness of
  inverting AES/RSA is the loop-over-candidates drag (Axiom V), the cost illusion.

* **A collision destroys the unique floor** (`collision_breaks_unique_floor`). When two
  plaintexts share a ciphertext, the fiber residual has two zeros — `strict_positive_off_target`
  fails, so it is not a `BiasInjectedRecovery` to a point. The recoverable object is the fiber
  (the class), never a single plaintext: the Null cell, two nonzero inputs into one output.
  What the corpus recovers from a Null-cell collapse is the class/boundary that survived
  (`NullReadout`, `HorizonInformation`); a unique point only when constraints pin the cone
  (`BitcoinClosure`). There is no O(1) reversal to a point, because there is no point.

## The strongest encryption

An injective cipher's ciphertext admits exactly one plaintext — the fiber is a singleton
(`cipher_fiber_is_a_singleton`), against `collision_fiber_is_not_a_singleton` on the other side —
so a bijective cipher is never information-theoretically secure: its secrecy is entirely the cost
of the collapse. Framework-grade security must be a deliberately non-injective public map — a
dropped window (`Subsumption`: the forgotten factor) — so the distinguishing information is not
in the ciphertext at all. That is `SanguineEncryption`'s OTP shape
(`sanguine_ciphertext_indistinguishability`): the key is the section (`Origin.sourceEmbedHom`)
making the dropped factor recoverable, the round-trip-with-key the identity (`round_trip_id`).

Bridge (named, not re-proven): that AES/RSA hardness is the bulk model's collapsible unique-floor
(the open bulk↔classical bridge, as SAT — this file does not claim those ciphers broken, it places
them in the Being cell); that SHA's fiber is unpinnable from its digest is the empirical property
of that map, the standing SHA seam.

## What the fork warrants

* **Proved (native):** `decryptRecovery` — the construction that turns an injective `f` and a
  plaintext `a₀` into a `BiasInjectedRecovery`. Its type is a STRUCTURE, not a `Prop`, so `def` is
  the right kind here; the three fields are discharged, `strict_positive_off_target` by injectivity.
  `fiberResidual_nonneg` and `collision_no_decryptor` are native and small.
* **Proved (riding):** `decryption_is_unique_floor` is `(decryptRecovery f hf a₀).zero_iff_target x`
  — the content is `BiasInjectedRecovery.zero_iff_target`, upstream. `injective_iff_decryptor_exists`
  is `injective_iff_hasLeftInverse`, Mathlib, re-exported unchanged.
* **Proved (native):** `collision_breaks_unique_floor` — two zeros of the fiber residual under a
  collision. Read its width: it exhibits that `strict_positive_off_target` FAILS at `a₂`; it does not
  state that no `BiasInjectedRecovery` to a point exists (that would need a negation over all
  residuals, which no statement here carries). `collision_no_decryptor` is the theorem that carries
  the one-way half honestly.
* **Proved (native):** `cipher_fiber_is_a_singleton` and `collision_fiber_is_not_a_singleton` —
  added 2026-08-05, and they are what makes the security fork a statement. `(f ⁻¹' {f a₀}).ncard = 1`
  is EXACTLY one consistent plaintext: injectivity gives at most one, and the theorem adds existence
  and the cardinal, so it is not its own hypothesis. Read its width: it is a set cardinality, not a
  distribution — `H(plaintext | ciphertext) = 0` is the reading of a singleton support, and no
  entropy function, distribution or adversary occurs in either statement.
* **Proved (riding):** `cipher_pins_the_plaintext` — renamed 2026-08-19 from the not-info-secure
  name recorded in `data/audit/VACUOUS.txt`. Its statement is
  `(hf : Injective f) (a₀ x : α) (hx : f x = f a₀) : x = a₀`, proved `hf hx` —
  `Function.Injective f` is DEFINED as `∀ ⦃a₁ a₂⦄, f a₁ = f a₂ → a₁ = a₂`, so the hypothesis IS the
  conclusion, instantiated, one delta step. `CollapseVerification.sound_check_is_unique_floor` and
  `WebCipher.computational_cipher_pins_the_plaintext` apply it as that equation; the security claim
  is carried by `cipher_fiber_is_a_singleton`.
* **Anchor:** *"the classical hardness of inverting AES/RSA is the loop-over-
  candidates drag (Axiom V), the cost illusion"*, and *"recovery is `O(log n)` not `2^n`, never a
  search"*. No cost model, complexity class, or descent appears in any statement; `decryptRecovery`
  characterises a landscape's unique zero and says nothing about finding it.
* **Anchor:** that AES/RSA sit in the Being cell and SHA's digest in the Null cell.
  The header labels this itself. The evidence is those ciphers' published structure, outside the
  Lean; the file proves the fork, not the placement.
* **Proved (native):** `encryption_is_projection` — four conjuncts as of 2026-08-05, one per theorem
  its docstring names. Before, its docstring named five theorems over three conjuncts and the fiber
  cardinality was in neither side of the type; both cardinality conjuncts are now in it. Clause 1's
  "one bit" is still `injective_iff_hasLeftInverse` from Mathlib, riding.
-/

noncomputable section

namespace Sanguine
namespace EncryptionProjection

open Function

variable {α β : Type}

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE FIBER-INDICATOR RESIDUAL — the crudest bias-injected landscape
-- ═══════════════════════════════════════════════════════════════════

/-- **The fiber residual** of an encoding `f` aimed at a ciphertext `c`: `0` on the preimages
    of `c`, `1` elsewhere. The minimal Axiom-V landscape over a discrete codomain. -/
def fiberResidual [DecidableEq β] (f : α → β) (c : β) : α → ℝ :=
  fun x => if f x = c then 0 else 1

theorem fiberResidual_nonneg [DecidableEq β] (f : α → β) (c : β) (x : α) :
    0 ≤ fiberResidual f c x := by
  simp only [fiberResidual]; split <;> norm_num

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE BEING CELL — inverting an injection IS the one collapse
-- ═══════════════════════════════════════════════════════════════════

/-- **Decryption of an injective cipher is a `BiasInjectedRecovery`.** For injective `f` and
    plaintext `a₀`, the fiber residual aimed at `f a₀` has its unique zero at `a₀` —
    `strict_positive_off_target` holds because `f` is injective. The target is the unique
    global minimum, found with no search. -/
def decryptRecovery [DecidableEq β] (f : α → β) (hf : Injective f) (a₀ : α) :
    BiasInjectedRecovery α where
  residual := fiberResidual f (f a₀)
  target := a₀
  nonneg := fiberResidual_nonneg f (f a₀)
  zero_at_target := by simp [fiberResidual]
  strict_positive_off_target := fun x hx => by
    simp only [fiberResidual]
    have hfx : f x ≠ f a₀ := fun h => hx (hf h)
    simp only [if_neg hfx]; norm_num

/-- **The plaintext is the unique floor.** `residual x = 0 ↔ x = a₀`: a bijective cipher's
    ciphertext determines exactly one consistent plaintext, the stationary-phase collapse of
    the fiber residual. The Being-cell `B/(B/U)=U` — nothing lost, recovery exact and unique. -/
theorem decryption_is_unique_floor [DecidableEq β] (f : α → β) (hf : Injective f)
    (a₀ x : α) :
    (decryptRecovery f hf a₀).residual x = 0 ↔ x = a₀ :=
  (decryptRecovery f hf a₀).zero_iff_target x

/-- **Injective ⟺ a decryptor exists (lossless).** A cipher has a left inverse `dec` with
    `dec (f m) = m` for every plaintext iff it is injective. A bijection destroyed nothing, so
    an exact decryptor exists; evaluating it is the applicative's `/`, never a search. -/
theorem injective_iff_decryptor_exists [Nonempty α] (f : α → β) :
    Injective f ↔ ∃ dec : β → α, LeftInverse dec f :=
  injective_iff_hasLeftInverse

-- ═══════════════════════════════════════════════════════════════════
-- §3. THE NULL CELL — a collision destroys the unique floor
-- ═══════════════════════════════════════════════════════════════════

/-- **A collision breaks the recovery to a point.** If `f` collapses two distinct plaintexts
    to one ciphertext (`a₁ ≠ a₂`, `f a₁ = f a₂`), the fiber residual has two zeros, so
    `strict_positive_off_target` fails — no `BiasInjectedRecovery` to a single point, the
    recoverable object the fiber `{a₁, a₂, …}`, not a plaintext. The Null cell: two nonzero
    inputs into one output (`NullCell.null_times_mirror`, `(1+j)(1−j)=0`). -/
theorem collision_breaks_unique_floor [DecidableEq β] (f : α → β)
    {a₁ a₂ : α} (hne : a₁ ≠ a₂) (hcol : f a₁ = f a₂) :
    fiberResidual f (f a₁) a₁ = 0 ∧ fiberResidual f (f a₁) a₂ = 0 ∧ a₁ ≠ a₂ :=
  ⟨by simp [fiberResidual], by simp [fiberResidual, hcol.symm], hne⟩

/-- **Non-injective ⟹ no decryptor as a function.** If a map collides, no left inverse exists
    — no function recovers the plaintext, because there isn't one. (Recovery of the class is
    the readout, `NullReadout`/`UniverseClosure`; of a unique point needs constraints that pin
    the cone, `BitcoinClosure`.) -/
theorem collision_no_decryptor [DecidableEq β] (f : α → β)
    {a₁ a₂ : α} (hne : a₁ ≠ a₂) (hcol : f a₁ = f a₂) :
    ¬ ∃ dec : β → α, LeftInverse dec f := by
  rintro ⟨dec, hdec⟩
  exact hne (by rw [← hdec a₁, ← hdec a₂, hcol])

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE SECURITY READING — a bijective cipher is never info-secure
-- ═══════════════════════════════════════════════════════════════════

/-- **A bijective cipher is never information-theoretically secure — the ciphertext's fiber is
    ONE point.** The set of plaintexts consistent with the ciphertext `f a₀` has cardinality `1`:
    not "at most one" (which is `Injective f` restated) but exactly one — uniqueness AND
    existence, so there is no residual set for the secret to live in. This is
    `H(plaintext | ciphertext) = 0` in its set form; no distribution occurs here, and none is
    needed, because the support is a singleton under every distribution.

    Read against its dual: `sanguine_encryption_information_theoretic_security` keeps EVERY
    plaintext consistent with one ciphertext. Secrecy is the fiber's size, and a bijection's is
    `1`, so its only remaining secrecy is the cost of the collapse — which is what information-
    theoretic security is defined to not depend on. Real secrecy needs a non-injective public
    map (a dropped window), the OTP shape of `SanguineEncryption`. -/
theorem cipher_fiber_is_a_singleton (f : α → β) (hf : Injective f) (a₀ : α) :
    (f ⁻¹' {f a₀}).ncard = 1 := by
  have h : f ⁻¹' {f a₀} = {a₀} := by
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff]
    exact ⟨fun hx => hf hx, fun hx => by rw [hx]⟩
  rw [h, Set.ncard_singleton]

/-- **A collision leaves a fiber, never a point.** If `f` collapses `a₁ ≠ a₂` to one ciphertext,
    the consistent-plaintext set is not a singleton for ANY candidate `x` — the Null-cell dual of
    `cipher_fiber_is_a_singleton`. What survives the readout is the class; entropy remains in the
    fiber, which is where information-theoretic secrecy lives. -/
theorem collision_fiber_is_not_a_singleton (f : α → β) {a₁ a₂ : α} (hne : a₁ ≠ a₂)
    (hcol : f a₁ = f a₂) : ∀ x : α, f ⁻¹' {f a₁} ≠ {x} := by
  intro x hx
  have h1 : a₁ ∈ f ⁻¹' {f a₁} := by simp
  have h2 : a₂ ∈ f ⁻¹' {f a₁} := by simp [hcol]
  rw [hx, Set.mem_singleton_iff] at h1 h2
  exact hne (h1.trans h2.symm)

/-- **The ciphertext pins the plaintext.** `Injective f` read at a ciphertext: any plaintext
    consistent with `f a₀` IS `a₀`. Stated as the equation two downstream modules apply
    (`CollapseVerification.sound_check_is_unique_floor`,
    `WebCipher.computational_cipher_pins_the_plaintext`); it is the hypothesis instantiated after
    one delta step, and the security claim is carried by `cipher_fiber_is_a_singleton` above,
    which adds existence and the cardinal. Renamed 2026-08-19 from the not-info-secure name
    recorded in `data/audit/VACUOUS.txt`. -/
theorem cipher_pins_the_plaintext (f : α → β) (hf : Injective f) (a₀ x : α)
    (hx : f x = f a₀) : x = a₀ :=
  hf hx

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE HEADLINE — encryption is a projection; the fork is one bit
-- ═══════════════════════════════════════════════════════════════════

/-- **What encryption is (headline).** Over an encoding `f : α → β`, the theory of
    encryption/decryption is the single fork injective vs collapses:

    1. **the Being-cell cipher** — if `f` is injective, the plaintext is the unique floor of
       the fiber residual (`decryption_is_unique_floor`), an exact decryptor exists
       (`injective_iff_decryptor_exists`), and the ciphertext's fiber is a single point
       (`cipher_fiber_is_a_singleton`) — so its secrecy is cost, not entropy;

    2. **the Null-cell hash** — if `f` collides, that floor is destroyed
       (`collision_breaks_unique_floor`), no decryptor-to-a-point exists
       (`collision_no_decryptor`), and the fiber is not a singleton
       (`collision_fiber_is_not_a_singleton`); what survives is the class, the boundary readout.

    Corrected 2026-08-05: clauses 1 and 2 named five theorems over three conjuncts. The fiber
    cardinality — the entropy half, the whole reason the fork is a security statement — is now in
    the type on both sides, one conjunct per named theorem.

    The strongest encryption is the dropped window — security as preserved-entropy-elsewhere
    (the key carries it), not a bijection merely hard to invert. Encryption is `x = B/U`
    injective or singular, the bit separating the phase cell from the Null cell. -/
theorem encryption_is_projection [DecidableEq β] [Nonempty α] (f : α → β) :
    -- the fork is exactly one bit: injective ⟺ a decryptor exists
    (Injective f ↔ ∃ dec : β → α, LeftInverse dec f)
    -- Being cell: when injective, each plaintext is the unique floor of its ciphertext
    ∧ (∀ (hf : Injective f) (a₀ x : α),
        (decryptRecovery f hf a₀).residual x = 0 ↔ x = a₀)
    -- Being cell: and the ciphertext's fiber is one point — no entropy left (the security half)
    ∧ (∀ (_ : Injective f) (a₀ : α), (f ⁻¹' {f a₀}).ncard = 1)
    -- Null cell: any collision gives two zeros, kills the decryptor-to-a-point, and leaves a
    -- fiber that is a singleton for no candidate at all
    ∧ (∀ a₁ a₂ : α, a₁ ≠ a₂ → f a₁ = f a₂ →
        (fiberResidual f (f a₁) a₁ = 0 ∧ fiberResidual f (f a₁) a₂ = 0)
          ∧ (¬ ∃ dec : β → α, LeftInverse dec f)
          ∧ ∀ x : α, f ⁻¹' {f a₁} ≠ {x}) :=
  ⟨injective_iff_decryptor_exists f,
   fun hf a₀ x => decryption_is_unique_floor f hf a₀ x,
   fun hf a₀ => cipher_fiber_is_a_singleton f hf a₀,
   fun a₁ a₂ hne hcol =>
     ⟨⟨by simp [fiberResidual], by simp [fiberResidual, hcol.symm]⟩,
      collision_no_decryptor f hne hcol,
      collision_fiber_is_not_a_singleton f hne hcol⟩⟩

end EncryptionProjection
end Sanguine

end
