import ApeironWord
import Denotation
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.Basic
import Apeiron.AuditClosed
import WordRefinement

/-!
# Face-bound refinement of the extracted cell kernels (obligation 4)

The port's obligation 4 (`/home/socol/oc-tmp/ht-port/receipt.md` §4) is the face-bound row of §3.1:
the *identity step* — that a cell is pinned by its readout — is stated with `σ < 0` in its
hypotheses, per `face_bound` (each theorem's hypotheses fix `σ`'s sign). The elliptic reading (the
multiplier is odd, hence a unit in `ZMod (2 ^ 64)`) is the identity step's admissible domain; it is
not a claim about `σ` alone, and it is **not** extended here to `σ = 0` or `σ > 0`.

The generated bodies are the real `apeiron_word.word_mul` (`proof/ApeironWord.lean:23983`, the
extracted `Cell<i64>::mul`) and `apeiron_word.word_norm` (`proof/ApeironWord.lean:23992`, the
extracted `Cell<i64>::norm_form`). `WordRefinement` already links both to the canonical
`Apeiron.Trichotomy.AnyRing.ringMul` / `ringNormForm` at `denoteSigma s`; this module adds the
elliptic specialisation and the canonical `σ < 0` face bound, and checks the corpus theorems it
rides (`normForm_eq_zero_iff_of_neg`, `isUnit_iff_normForm_ne_zero`,
`eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`, `noZeroDivisors_iff_disc_neg`,
`AnyRing.exists_inverse_iff_isUnit_ringNormForm`, `zeroDivisorPair_of_disc_one`,
`nilpotent_of_disc_zero`) rather than citing them.

The bound's exact content, over `ZMod (2 ^ 64)`: the divisors there are the **odd** words, so the
word-ring identity step is `IsUnit (ringNormForm (-1) z)`, not `ringNormForm (-1) z ≠ 0`. The
canonical `ℝ` statement `normForm σ z = 0 ↔ z = 0` has no verbatim `ZMod` transport; only the unit
criterion does, and that is what is proved. `hyperbolic_null_products` and `parabolic_nilpotent`
exhibit the failure above and at zero.
-/

namespace Apeiron.GrowableFaceBound

open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.Trichotomy

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

/-! ## The dial values in the word ring -/

/-- The machine word `-1` denotes `-1` in `ZMod (2 ^ 64)`. -/
theorem word_denote_neg_one : denote (-1)#i64 = (-1 : ZMod (2 ^ 64)) := by
  have h : (-1)#i64 = core.num.I64.wrapping_sub 0#i64 1#i64 := by decide
  rw [h, denote_wrapping_sub, denote_zero, denote_one]
  ring

/-- The elliptic dial denotes `-1`, an odd word and a unit of the word ring. -/
theorem denoteSigma_elliptic : denoteSigma cell.Sigma.Elliptic = (-1 : ZMod (2 ^ 64)) := by
  simp only [denoteSigma, sigmaI64]
  exact word_denote_neg_one

/-- The parabolic dial denotes `0`, an even word and not a unit. -/
theorem denoteSigma_parabolic : denoteSigma cell.Sigma.Parabolic = (0 : ZMod (2 ^ 64)) := by
  simp [denoteSigma, sigmaI64]

/-- The elliptic dial's word is a unit of `ZMod (2 ^ 64)`: the odd multiplier the identity step
needs. This is the admissible-domain condition, and it is a fact about the multiplier, not a sign
claim about `σ`. -/
theorem elliptic_dial_isUnit : IsUnit (denoteSigma cell.Sigma.Elliptic) := by
  rw [denoteSigma_elliptic]
  exact isUnit_neg_one

/-- The parabolic dial's even word is not a unit, so the same multiplier reading refuses the middle
face. -/
theorem parabolic_dial_not_isUnit : ¬ IsUnit (denoteSigma cell.Sigma.Parabolic) := by
  haveI : Fact (1 < 2 ^ 64) := ⟨by norm_num⟩
  rw [denoteSigma_parabolic]
  exact fun h => h.ne_zero rfl

/-! ## The generated bodies at the elliptic dial -/

/-- The extracted `word_mul` at the elliptic dial refines `Apeiron.Trichotomy.AnyRing.ringMul` at
`σ = -1`. `WordRefinement.word_mul_denote` is the all-`σ` link; this is its elliptic face. -/
theorem word_mul_denote_elliptic (a b : cell.Cell I64) :
    (do let c ← word_mul a cell.Sigma.Elliptic b; ok (denoteCell c)) =
      ok (AnyRing.ringMul (-1 : ZMod (2 ^ 64)) (denoteCell a) (denoteCell b)) := by
  rw [word_mul_denote, denoteSigma_elliptic]

/-- The extracted `word_norm` at the elliptic dial refines
`Apeiron.Trichotomy.AnyRing.ringNormForm` at `σ = -1`. `WordRefinement.word_norm_denote` is the
all-`σ` link; this is its elliptic face. -/
theorem word_norm_denote_elliptic (a : cell.Cell I64) :
    (do let r ← word_norm a cell.Sigma.Elliptic; ok (denote r)) =
      ok (AnyRing.ringNormForm (-1 : ZMod (2 ^ 64)) (denoteCell a)) := by
  rw [word_norm_denote, denoteSigma_elliptic]

/-! ## The canonical elliptic face bound, `σ < 0` -/

/-- The face bound, checked from the corpus: below zero the canonical norm form vanishes only at
the origin. This is `Apeiron.Trichotomy.normForm_eq_zero_iff_of_neg` at the canonical
`AnyRing.ringNormForm` on `RingCell ℝ`; `σ < 0` is the hypothesis `face_bound` requires. -/
theorem elliptic_normForm_eq_zero_iff {σ : ℝ} (hσ : σ < 0) (z : AnyRing.RingCell ℝ) :
    AnyRing.ringNormForm σ z = 0 ↔ z = 0 :=
  Apeiron.Trichotomy.normForm_eq_zero_iff_of_neg hσ z

/-- The elliptic identity step: with `σ < 0` a canonical cell is invertible exactly when it is
nonzero, so its norm form is a faithful coordinate. This is the canonical
`isUnit_iff_normForm_ne_zero` read through the face bound; it is stated only on this face. -/
theorem elliptic_identityStep {σ : ℝ} (hσ : σ < 0) (z : AnyRing.RingCell ℝ) :
    (∃ w, AnyRing.ringMul σ z w = AnyRing.ringOne) ↔ z ≠ 0 := by
  rw [Apeiron.Trichotomy.isUnit_iff_normForm_ne_zero σ z]
  exact not_congr (elliptic_normForm_eq_zero_iff hσ z)

/-- Below zero the canonical cell ring has no zero divisors: a zero product forces a zero factor.
Checked from `eq_zero_or_eq_zero_of_mul_eq_zero_of_neg`. -/
theorem elliptic_no_zero_divisors {σ : ℝ} (hσ : σ < 0) (z w : AnyRing.RingCell ℝ)
    (h : AnyRing.ringMul σ z w = 0) : z = 0 ∨ w = 0 :=
  Apeiron.Trichotomy.eq_zero_or_eq_zero_of_mul_eq_zero_of_neg hσ z w h

/-- The classification, checked from `noZeroDivisors_iff_disc_neg`: only below zero does the
canonical cell ring have no zero divisors. -/
theorem noZeroDivisors_iff_disc_neg (σ : ℝ) :
    (∀ z w : AnyRing.RingCell ℝ, AnyRing.ringMul σ z w = 0 → z = 0 ∨ w = 0) ↔ σ < 0 :=
  Apeiron.Trichotomy.noZeroDivisors_iff_disc_neg σ

/-! ## The word-ring identity step: the unit criterion -/

/-- The word-ring identity step at the elliptic dial: a cell has a `ringMul (-1)` inverse exactly
when its norm form is a unit of `ZMod (2 ^ 64)` — an odd word. This is the `Z/2^64` reading of
`isUnit_iff_normForm_ne_zero`: the divisors there are the odd elements, not the nonzero ones, so
`≠ 0` is not the criterion. Checked from `AnyRing.exists_inverse_iff_isUnit_ringNormForm`. -/
theorem word_elliptic_invertible_iff_isUnit_norm (z : AnyRing.RingCell (ZMod (2 ^ 64))) :
    (∃ w, AnyRing.ringMul (-1 : ZMod (2 ^ 64)) z w = AnyRing.ringOne) ↔
      IsUnit (AnyRing.ringNormForm (-1 : ZMod (2 ^ 64)) z) :=
  Apeiron.Trichotomy.AnyRing.exists_inverse_iff_isUnit_ringNormForm (-1) z

/-! ## The bound is not claimed above zero or at zero -/

/-- Above zero the identity step fails: at `σ = 1` two nonzero canonical cells multiply to zero.
Checked from `zeroDivisorPair_of_disc_one`. -/
theorem hyperbolic_null_products :
    AnyRing.ringMul (1 : ℝ) (1, 1) (1, -1) = 0 ∧
      ((1, 1) : AnyRing.RingCell ℝ) ≠ 0 ∧ ((1, -1) : AnyRing.RingCell ℝ) ≠ 0 :=
  Apeiron.Trichotomy.zeroDivisorPair_of_disc_one

/-- At `σ = 0` a nonzero canonical cell squares to zero, so the parabolic face is not the identity
step's domain either. Checked from `nilpotent_of_disc_zero`. -/
theorem parabolic_nilpotent :
    AnyRing.ringMul (0 : ℝ) (0, 1) (0, 1) = 0 ∧ ((0, 1) : AnyRing.RingCell ℝ) ≠ 0 :=
  Apeiron.Trichotomy.nilpotent_of_disc_zero

/-! ## The headline: the assembled refinement -/

/-- **The elliptic face bound.** The port's obligation-4 refinement, assembled. The extracted
`word_mul` / `word_norm` bodies at the elliptic dial refine the canonical `AnyRing.ringMul` /
`ringNormForm` at `σ = -1`; the canonical `ℝ` identity step and no-zero-divisor statements are
stated with `σ < 0`; the word-ring identity step is the *unit* criterion; and the dial word is a
unit. Every field carries its `σ`-sign hypothesis where `face_bound` requires one, and no field
claims the statement for `σ = 0` or `σ > 0`. -/
structure EllipticFaceBound : Prop where
  mul_refines : ∀ a b : cell.Cell I64,
    (do let c ← word_mul a cell.Sigma.Elliptic b; ok (denoteCell c)) =
      ok (AnyRing.ringMul (-1 : ZMod (2 ^ 64)) (denoteCell a) (denoteCell b))
  norm_refines : ∀ a : cell.Cell I64,
    (do let r ← word_norm a cell.Sigma.Elliptic; ok (denote r)) =
      ok (AnyRing.ringNormForm (-1 : ZMod (2 ^ 64)) (denoteCell a))
  real_norm_vanishes_iff : ∀ {σ : ℝ}, σ < 0 → ∀ z : AnyRing.RingCell ℝ,
    AnyRing.ringNormForm σ z = 0 ↔ z = 0
  real_identity_step : ∀ {σ : ℝ}, σ < 0 → ∀ z : AnyRing.RingCell ℝ,
    (∃ w, AnyRing.ringMul σ z w = AnyRing.ringOne) ↔ z ≠ 0
  real_no_zero_divisors : ∀ {σ : ℝ}, σ < 0 → ∀ z w : AnyRing.RingCell ℝ,
    AnyRing.ringMul σ z w = 0 → z = 0 ∨ w = 0
  word_identity_step : ∀ z : AnyRing.RingCell (ZMod (2 ^ 64)),
    (∃ w, AnyRing.ringMul (-1 : ZMod (2 ^ 64)) z w = AnyRing.ringOne) ↔
      IsUnit (AnyRing.ringNormForm (-1 : ZMod (2 ^ 64)) z)
  dial_unit : IsUnit (denoteSigma cell.Sigma.Elliptic)

/-- The port's face-bound refinement is realized. -/
theorem elliptic_faceBound : EllipticFaceBound :=
  { mul_refines := word_mul_denote_elliptic
    norm_refines := word_norm_denote_elliptic
    real_norm_vanishes_iff := fun hσ z => elliptic_normForm_eq_zero_iff hσ z
    real_identity_step := fun hσ z => elliptic_identityStep hσ z
    real_no_zero_divisors := fun hσ z w h => elliptic_no_zero_divisors hσ z w h
    word_identity_step := word_elliptic_invertible_iff_isUnit_norm
    dial_unit := elliptic_dial_isUnit }

#audit Apeiron.GrowableFaceBound.elliptic_faceBound
#audit Apeiron.GrowableFaceBound.elliptic_identityStep
#audit Apeiron.GrowableFaceBound.elliptic_normForm_eq_zero_iff
#audit Apeiron.GrowableFaceBound.word_elliptic_invertible_iff_isUnit_norm
#audit Apeiron.GrowableFaceBound.word_norm_denote_elliptic
#audit Apeiron.GrowableFaceBound.word_mul_denote_elliptic

end Apeiron.GrowableFaceBound
