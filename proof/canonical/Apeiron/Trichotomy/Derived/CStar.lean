import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.Idempotents
import Apeiron.Trichotomy.NullBasis

/-!
# C*-Algebraic and Operator Structures of the σ-Trichotomy

@name CStar
@cite cockle-1849
@cite peirce-1870
@cite boole-1847
@cite euclid-elements-ii-5
@grade proved

This module formally proves the correspondence between the Apeiron σ-trichotomy
`A_σ = ℝ[x]/(x² − σ)` and operator-algebraic structures:

1. **Hyperbolic sector (σ > 0):**
   - The carrier `Cell` under the null basis `upper`/`lower` and sup-norm is a
     commutative real C*-algebra under diagonal involution `starDiag z = z`.
   - The C*-identity `‖z* z‖ = ‖z‖²` holds (`cstar_identity_pos`), and `1 + z* z`
     is invertible for every element (`one_add_star_mul_self_isUnit`).
   - The canonical involution `conj` is an indefinite Krein involution, failing
     the C*-identity on the null idempotents (`conj_not_cstar_pos`).
   - The fundamental symmetry `J` relates the diagonal C*-involution and `conj`
     via `J M_z J = M_{conj z}` (`krein_adjoint_operator`, `krein_adjoint_eq_conj`).

2. **Parabolic sector (σ = 0):**
   - The dual numbers carry a non-trivial nilpotent radical (`jacobson_radical_zero`).
   - No norm and involution on `Cell` at σ = 0 can satisfy the C*-identity
     (`no_cstar_norm_zero`), representing the Inönü-Wigner contraction.

3. **Spectral Multiplier and Channel Decomposition:**
   - For `normForm σ w ≠ 0`, multiplication by the reciprocal `recip σ w` acts as
     a coordinate-wise quotient on each spectral channel (`ratio_spectral_representation`).
   - The double reciprocal multiplication inverts back to `w` (`recovery_by_rotation_cell`),
     realizing modular reflexivity.

4. **Elliptic sector (σ < 0):**
   - The carrier `Cell` under the positive-definite norm form `cstarNormNeg σ z = √(normForm σ z)`
     forms a 2-dimensional real C*-algebra under canonical conjugation `conj z`.
   - The norm is non-negative (`cstarNormNeg_nonneg`), non-degenerate (`cstarNormNeg_eq_zero_iff`),
     and multiplicative (`cstarNormNeg_mul`).
   - The C*-identity `‖z* z‖ = ‖z‖²` holds (`cstar_identity_neg`), and `1 + z* z`
     is invertible for every element (`one_add_star_mul_self_isUnit_neg`).

5. **Unified Trichotomy C*-Classification:**
   - Synthesizes the tripartite division across elliptic, hyperbolic, and parabolic
     sectors into a single theorem (`trichotomy_cstar_classification`).
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

open scoped Real

-- ═══════════════════════════════════════════════════════════════════
-- §1. HYPERBOLIC SECTOR (σ > 0): REAL C*-ALGEBRA STRUCTURE
-- ═══════════════════════════════════════════════════════════════════

/-- The diagonal C*-involution on Cell at σ > 0.

    (z : Cell)                 [a point]
    : Cell                     [the point itself]

    In the idempotent decomposition ℝ ⊕ ℝ, the standard C*-involution is
    the identity map. -/
def starDiag (z : Cell) : Cell := z

/-- The C*-norm on Cell at σ > 0: the L^∞ (supremum) norm in the null basis.

    (σ : ℝ)                    [the parameter]
    (z : Cell)                 [the point being measured]
    : ℝ                        [the maximum absolute coordinate] -/
def cstarNormPos (σ : ℝ) (z : Cell) : ℝ :=
  max |upper σ z| |lower σ z|

/-- Linear distributivity of the upper channel over vector addition.

    (σ : ℝ)                    [the parameter]
    (z w : Cell)               [two points]
    upper σ (z + w) = upper σ z + upper σ w  [distributes over sum] -/
theorem upper_add (σ : ℝ) (z w : Cell) :
    upper σ (z + w) = upper σ z + upper σ w := by
  simp only [upper, Prod.fst_add, Prod.snd_add]
  ring

/-- Linear distributivity of the lower channel over vector addition.

    (σ : ℝ)                    [the parameter]
    (z w : Cell)               [two points]
    lower σ (z + w) = lower σ z + lower σ w  [distributes over sum] -/
theorem lower_add (σ : ℝ) (z w : Cell) :
    lower σ (z + w) = lower σ z + lower σ w := by
  simp only [lower, Prod.fst_add, Prod.snd_add]
  ring

/-- The C*-identity holds for starDiag and cstarNormPos at σ > 0.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [any point]
    mul                        [multiply two points together, the σ way]
    cstarNormPos σ (mul σ (starDiag z) z) = (cstarNormPos σ z) ^ 2
                               [norm of z* z is norm of z squared] -/
theorem cstar_identity_pos (σ : ℝ) (hσ : 0 < σ) (z : Cell) :
    cstarNormPos σ (mul σ (starDiag z) z) = (cstarNormPos σ z) ^ 2 := by
  unfold starDiag cstarNormPos
  rw [upper_mul hσ.le, lower_mul hσ.le]
  rw [abs_mul, abs_mul]
  have hu : |upper σ z| * |upper σ z| = |upper σ z| ^ 2 := (sq (|upper σ z|)).symm
  have hl : |lower σ z| * |lower σ z| = |lower σ z| ^ 2 := (sq (|lower σ z|)).symm
  rw [hu, hl]
  rcases le_total (|upper σ z|) (|lower σ z|) with hle | hle
  · rw [max_eq_right hle]
    have hsq : |upper σ z| ^ 2 ≤ |lower σ z| ^ 2 := by
      rw [sq, sq]
      exact mul_self_le_mul_self (abs_nonneg (upper σ z)) hle
    rw [max_eq_right hsq]
  · rw [max_eq_left hle]
    have hsq : |lower σ z| ^ 2 ≤ |upper σ z| ^ 2 := by
      rw [sq, sq]
      exact mul_self_le_mul_self (abs_nonneg (lower σ z)) hle
    rw [max_eq_left hsq]

/-- In the real C*-algebra, 1 + z* z is strictly positive on both channels,
    hence invertible for all z at σ > 0.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [a point: two real numbers written side by side, a + b·x]
    one                        [the point (1, 0), which changes nothing when you multiply by it]
    mul                        [multiply two points together, the σ way]
    ∃ w : Cell                 [there is at least one]
    mul σ (one + mul σ (starDiag z) z) w = one
                               [1 + z* z has a multiplicative inverse] -/
theorem one_add_star_mul_self_isUnit (σ : ℝ) (hσ : 0 < σ) (z : Cell) :
    ∃ w : Cell, mul σ (one + mul σ (starDiag z) z) w = one := by
  unfold starDiag
  set v := one + mul σ z z
  have hu : upper σ v = 1 + (upper σ z) ^ 2 := by
    rw [upper_add, (halves_one σ).1, upper_mul hσ.le, sq]
  have hl : lower σ v = 1 + (lower σ z) ^ 2 := by
    rw [lower_add, (halves_one σ).2, lower_mul hσ.le, sq]
  have hnorm : normForm σ v = (1 + (upper σ z) ^ 2) * (1 + (lower σ z) ^ 2) := by
    rw [normForm_eq_upper_mul_lower hσ.le v, hu, hl]
  have hu_pos : 0 < 1 + (upper σ z) ^ 2 := by
    have : 0 ≤ (upper σ z) ^ 2 := sq_nonneg _
    linarith
  have hl_pos : 0 < 1 + (lower σ z) ^ 2 := by
    have : 0 ≤ (lower σ z) ^ 2 := sq_nonneg _
    linarith
  have hnorm_pos : 0 < normForm σ v := by
    rw [hnorm]
    exact mul_pos hu_pos hl_pos
  have hnorm_ne : normForm σ v ≠ 0 := ne_of_gt hnorm_pos
  exact (isUnit_iff_normForm_ne_zero σ v).mpr hnorm_ne

-- ═══════════════════════════════════════════════════════════════════
-- §2. KREIN INVOLUTION AND INDEFINITE METRIC
-- ═══════════════════════════════════════════════════════════════════

/-- Conjugation swaps the positive and negative idempotents.

    (σ : ℝ)                    [the parameter]
    conj (idemPlus σ) = idemMinus σ  [conjugation swaps the idempotents] -/
theorem conj_idemPlus (σ : ℝ) : conj (idemPlus σ) = idemMinus σ := by
  simp only [conj, idemPlus, idemMinus]

/-- The positive and negative idempotents are mutually orthogonal.

    {σ : ℝ} (hσ : 0 < σ)       [the parameter, strictly above zero]
    mul σ (idemMinus σ) (idemPlus σ) = 0  [their product is zero] -/
theorem mul_idemMinus_idemPlus {σ : ℝ} (hσ : 0 < σ) :
    mul σ (idemMinus σ) (idemPlus σ) = 0 := by
  apply halves_ext hσ
  · rw [upper_mul hσ.le, (halves_idemMinus hσ).1, zero_mul, (halves_zero σ).1]
  · rw [lower_mul hσ.le, (halves_idemPlus hσ).2, mul_zero, (halves_zero σ).2]

/-- The canonical involution conj is NOT a C*-involution for σ > 0:
    the C*-identity fails on the non-trivial null idempotent.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [a point: two real numbers written side by side, a + b·x]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    ¬                          [this is not true]
    ∀                          [for every single one]
    ¬ (∀ z : Cell, cstarNormPos σ (mul σ (conj z) z) = (cstarNormPos σ z) ^ 2)
                               [the C*-identity fails under conj] -/
theorem conj_not_cstar_pos (σ : ℝ) (hσ : 0 < σ) :
    ¬ (∀ z : Cell, cstarNormPos σ (mul σ (conj z) z) = (cstarNormPos σ z) ^ 2) := by
  intro h_cstar
  have h_idem := h_cstar (idemPlus σ)
  rw [conj_idemPlus σ, mul_idemMinus_idemPlus hσ] at h_idem
  have h_norm_zero : cstarNormPos σ 0 = 0 := by
    unfold cstarNormPos
    rw [(halves_zero σ).1, (halves_zero σ).2, abs_zero, max_self]
  have h_norm_idem : cstarNormPos σ (idemPlus σ) = 1 := by
    unfold cstarNormPos
    rw [(halves_idemPlus hσ).1, (halves_idemPlus hσ).2, abs_one, abs_zero]
    exact max_eq_left (by norm_num : (0 : ℝ) ≤ 1)
  rw [h_norm_zero, h_norm_idem, one_pow] at h_idem
  norm_num at h_idem

/-- The fundamental symmetry J on Cell, interchanging upper and lower channels.
    This is the metric operator relating the positive Hilbert inner product
    and the indefinite Krein form of signature (1, 1).

    (σ : ℝ)                    [the parameter]
    (z : Cell)                 [a point]
    : Cell                     [the point with channels exchanged] -/
def fundamentalSymmetry (σ : ℝ) (z : Cell) : Cell :=
  ofHalves σ (lower σ z) (upper σ z)

/-- The fundamental symmetry coincides with canonical conjugation.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [a point]
    fundamentalSymmetry σ z = conj z  [the symmetry matches conjugation] -/
theorem fundamentalSymmetry_eq_conj (σ : ℝ) (hσ : 0 < σ) (z : Cell) :
    fundamentalSymmetry σ z = conj z := by
  have hsne : Real.sqrt σ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hσ)
  obtain ⟨a, b⟩ := z
  simp only [fundamentalSymmetry, ofHalves, upper, lower, conj, Prod.ext_iff]
  constructor
  · ring
  · field_simp; ring

/-- The fundamental symmetry is involutive (J² = I).

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [a point]
    fundamentalSymmetry σ (fundamentalSymmetry σ z) = z  [applying twice returns z] -/
theorem fundamentalSymmetry_involutive (σ : ℝ) (hσ : 0 < σ) (z : Cell) :
    fundamentalSymmetry σ (fundamentalSymmetry σ z) = z := by
  rw [fundamentalSymmetry_eq_conj σ hσ, fundamentalSymmetry_eq_conj σ hσ, conj_conj]

/-- The Krein adjoint of multiplication by z coincides with multiplication
    by conj z: J M_z J = M_{conj z}.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z w : Cell)               [two points]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    fundamentalSymmetry σ (mul σ (starDiag z) (fundamentalSymmetry σ w)) = mul σ (conj z) w
                               [operator equation J M_z J = M_{conj z}] -/
theorem krein_adjoint_operator (σ : ℝ) (hσ : 0 < σ) (z w : Cell) :
    fundamentalSymmetry σ (mul σ (starDiag z) (fundamentalSymmetry σ w)) = mul σ (conj z) w := by
  unfold starDiag
  rw [fundamentalSymmetry_eq_conj σ hσ w]
  have h1 : fundamentalSymmetry σ (mul σ z (conj w)) = conj (mul σ z (conj w)) :=
    fundamentalSymmetry_eq_conj σ hσ _
  rw [h1]
  obtain ⟨a, b⟩ := z
  obtain ⟨c, d⟩ := w
  simp only [mul, conj, Prod.ext_iff]
  constructor <;> ring

/-- Krein adjoint evaluated on the identity yields conj z.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z : Cell)                 [a point: two real numbers written side by side, a + b·x]
    one                        [the point (1, 0), which changes nothing when you multiply by it]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    fundamentalSymmetry σ (mul σ (starDiag z) (fundamentalSymmetry σ one)) = conj z
                               [action on the unit is conjugation] -/
theorem krein_adjoint_eq_conj (σ : ℝ) (hσ : 0 < σ) (z : Cell) :
    fundamentalSymmetry σ (mul σ (starDiag z) (fundamentalSymmetry σ one)) = conj z := by
  rw [krein_adjoint_operator σ hσ z one, mul_comm, one_mul]

-- ═══════════════════════════════════════════════════════════════════
-- §3. PARABOLIC SECTOR (σ = 0): NILPOTENT RADICAL AND NO C*-NORM
-- ═══════════════════════════════════════════════════════════════════

/-- The Jacobson radical of Cell at σ = 0 contains a non-zero nilpotent.

    ∃ z : Cell                 [there is at least one point]
    z ≠ 0                      [not the origin]
    ∧                          [both are true]
    mul 0 z z = 0              [its square is the origin] -/
theorem jacobson_radical_zero :
    ∃ z : Cell, z ≠ 0 ∧ mul 0 z z = 0 :=
  ⟨(0, 1), nilpotent_of_disc_zero.2, nilpotent_of_disc_zero.1⟩

/-- No norm on Cell at σ = 0 can satisfy the C*-identity.
    In any C*-algebra, nilpotent elements have vanishing norm, which contradicts
    positive-definiteness on (0, 1).

    (norm : Cell → ℝ)          [any candidate norm]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    ∀                          [for every single one]
    (h_cstar : ∀ z : Cell, norm (mul 0 (conj z) z) = (norm z) ^ 2)
                               [satisfying the C*-identity]
    ↔                          [exactly when, true in both directions]
    (h_zero : ∀ z : Cell, norm z = 0 ↔ z = 0)
                               [vanishing only at zero]
    False                      [leads to a contradiction] -/
theorem no_cstar_norm_zero (norm : Cell → ℝ)
    (h_cstar : ∀ z : Cell, norm (mul 0 (conj z) z) = (norm z) ^ 2)
    (h_zero : ∀ z : Cell, norm z = 0 ↔ z = 0) : False := by
  have hz_ne : ((0, 1) : Cell) ≠ 0 := by
    intro h; have := congrArg Prod.snd h; norm_num at this
  have h_mul_zero : mul 0 (conj ((0, 1) : Cell)) (0, 1) = 0 := by
    simp only [conj, mul, Prod.ext_iff, Prod.fst_zero, Prod.snd_zero]
    norm_num
  have h_sq : (norm (0, 1)) ^ 2 = norm (0 : Cell) := by
    rw [← h_cstar (0, 1), h_mul_zero]
  have h_norm_zero : norm (0 : Cell) = 0 := by
    rw [h_zero]
  rw [h_norm_zero] at h_sq
  have h_norm_nil : norm (0, 1) = 0 := sq_eq_zero_iff.mp h_sq
  have hz_eq : ((0, 1) : Cell) = 0 := (h_zero (0, 1)).mp h_norm_nil
  exact hz_ne hz_eq

-- ═══════════════════════════════════════════════════════════════════
-- §4. SPECTRAL MULTIPLIERS AND CHANNEL DECOMPOSITION
-- ═══════════════════════════════════════════════════════════════════

/-- The upper channel of the reciprocal is the inverse of the upper channel.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (w : Cell)                 [the point being inverted]
    (hw : normForm σ w ≠ 0)    [lies off the cone]
    upper σ (recip σ w) = (upper σ w)⁻¹  [reciprocal on upper channel] -/
theorem upper_recip (σ : ℝ) (hσ : 0 < σ) (w : Cell) (hw : normForm σ w ≠ 0) :
    upper σ (recip σ w) = (upper σ w)⁻¹ := by
  have hnorm : normForm σ w = upper σ w * lower σ w := normForm_eq_upper_mul_lower hσ.le w
  have hu : upper σ w ≠ 0 := by
    intro h; rw [hnorm, h, zero_mul] at hw; exact hw rfl
  have hl : lower σ w ≠ 0 := by
    intro h; rw [hnorm, h, mul_zero] at hw; exact hw rfl
  simp only [upper, recip]
  rw [hnorm]
  have hcomb : w.1 / (upper σ w * lower σ w) + Real.sqrt σ * (-w.2 / (upper σ w * lower σ w))
      = (w.1 - Real.sqrt σ * w.2) / (upper σ w * lower σ w) := by ring
  rw [hcomb]
  show lower σ w / (upper σ w * lower σ w) = (upper σ w)⁻¹
  field_simp

/-- The lower channel of the reciprocal is the inverse of the lower channel.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (w : Cell)                 [the point being inverted]
    (hw : normForm σ w ≠ 0)    [lies off the cone]
    lower σ (recip σ w) = (lower σ w)⁻¹  [reciprocal on lower channel] -/
theorem lower_recip (σ : ℝ) (hσ : 0 < σ) (w : Cell) (hw : normForm σ w ≠ 0) :
    lower σ (recip σ w) = (lower σ w)⁻¹ := by
  have hnorm : normForm σ w = upper σ w * lower σ w := normForm_eq_upper_mul_lower hσ.le w
  have hu : upper σ w ≠ 0 := by
    intro h; rw [hnorm, h, zero_mul] at hw; exact hw rfl
  have hl : lower σ w ≠ 0 := by
    intro h; rw [hnorm, h, mul_zero] at hw; exact hw rfl
  simp only [lower, recip]
  rw [hnorm]
  have hcomb : w.1 / (upper σ w * lower σ w) - Real.sqrt σ * (-w.2 / (upper σ w * lower σ w))
      = (w.1 + Real.sqrt σ * w.2) / (upper σ w * lower σ w) := by ring
  rw [hcomb]
  show upper σ w / (upper σ w * lower σ w) = (lower σ w)⁻¹
  field_simp

/-- Spectral representation of division/multiplier by reciprocal when w is off the cone.
    Under the Gelfand isomorphism A_σ ≅ ℝ ⊕ ℝ, the product with reciprocal
    is given by coordinate-wise division on each spectral channel.

    (σ : ℝ) (hσ : 0 < σ)       [the parameter, strictly above zero]
    (z w : Cell)               [two points]
    (hw : normForm σ w ≠ 0)    [point w is off the null cone]
    upper σ (mul σ z (recip σ w)) = upper σ z / upper σ w   [ratio along the upper direction]
    ∧                          [both are true]
    lower σ (mul σ z (recip σ w)) = lower σ z / lower σ w   [ratio along the lower direction] -/
theorem ratio_spectral_representation (σ : ℝ) (hσ : 0 < σ) (z w : Cell)
    (hw : normForm σ w ≠ 0) :
    upper σ (mul σ z (recip σ w)) = upper σ z / upper σ w ∧
    lower σ (mul σ z (recip σ w)) = lower σ z / lower σ w := by
  constructor
  · rw [upper_mul hσ.le, upper_recip σ hσ w hw, div_eq_mul_inv]
  · rw [lower_mul hσ.le, lower_recip σ hσ w hw, div_eq_mul_inv]

/-- Recovery for Cell multipliers off the cone:
    z * (z * w⁻¹)⁻¹ = w.
    In operator theory, this is the reflexivity of relative modular operators.

    (σ : ℝ)                    [the parameter]
    (z w : Cell)               [two points]
    (hz : normForm σ z ≠ 0)    [point z is off the null cone]
    (hw : normForm σ w ≠ 0)    [point w is off the null cone]
    mul σ z (recip σ (mul σ z (recip σ w))) = w   [multiply two points together, the σ way] -/
theorem recovery_by_rotation_cell (σ : ℝ) (z w : Cell)
    (hz : normForm σ z ≠ 0) (hw : normForm σ w ≠ 0) :
    mul σ z (recip σ (mul σ z (recip σ w))) = w := by
  have hX_norm : normForm σ (mul σ z (recip σ w)) ≠ 0 := by
    rw [normForm_mul, normForm_recip σ w hw]
    exact mul_ne_zero hz (inv_ne_zero hw)
  have hXw : mul σ (mul σ z (recip σ w)) w = z := by
    calc mul σ (mul σ z (recip σ w)) w
        = mul σ z (mul σ (recip σ w) w) := mul_assoc σ z _ w
      _ = mul σ z (mul σ w (recip σ w)) := by rw [mul_comm σ (recip σ w) w]
      _ = mul σ z one := by rw [mul_recip_cancel σ w hw]
      _ = z := by rw [mul_comm σ z one, one_mul]
  calc mul σ z (recip σ (mul σ z (recip σ w)))
      = mul σ (recip σ (mul σ z (recip σ w))) z := mul_comm σ _ _
    _ = mul σ (recip σ (mul σ z (recip σ w))) (mul σ (mul σ z (recip σ w)) w) := by rw [hXw]
    _ = mul σ (mul σ (recip σ (mul σ z (recip σ w))) (mul σ z (recip σ w))) w :=
      (mul_assoc σ _ _ w).symm
    _ = mul σ (mul σ (mul σ z (recip σ w)) (recip σ (mul σ z (recip σ w)))) w := by
      rw [mul_comm σ (recip σ _) _]
    _ = mul σ one w := by rw [mul_recip_cancel σ (mul σ z (recip σ w)) hX_norm]
    _ = w := one_mul σ w


-- ═══════════════════════════════════════════════════════════════════
-- §5. ELLIPTIC SECTOR (σ < 0): DEFINITE REAL C*-ALGEBRA STRUCTURE
-- ═══════════════════════════════════════════════════════════════════

/-- The norm form is non-negative when σ ≤ 0.

    {σ : ℝ} (hσ : σ ≤ 0)       [σ is at zero or below]
    (z : Cell)                 [a point]
    0 ≤ normForm σ z           [size is non-negative] -/
theorem normForm_nonneg_of_nonpos {σ : ℝ} (hσ : σ ≤ 0) (z : Cell) :
    0 ≤ normForm σ z := by
  have ha : 0 ≤ z.1 ^ 2 := sq_nonneg z.1
  have hb : 0 ≤ -σ * z.2 ^ 2 := mul_nonneg (neg_nonneg.mpr hσ) (sq_nonneg z.2)
  have : normForm σ z = z.1 ^ 2 + (-σ * z.2 ^ 2) := by unfold normForm; ring
  rw [this]
  exact add_nonneg ha hb

/-- Conjugation preserves the norm form.

    (σ : ℝ) (z : Cell)         [the parameter and a point]
    normForm σ (conj z) = normForm σ z  [conjugate has same size] -/
theorem normForm_conj (σ : ℝ) (z : Cell) : normForm σ (conj z) = normForm σ z := by
  unfold normForm conj; ring

/-- The C*-norm on Cell at σ < 0: the square root of the positive-definite norm form.

    (σ : ℝ)                    [the parameter]
    (z : Cell)                 [the point being measured]
    : ℝ                        [the square root of its size] -/
def cstarNormNeg (σ : ℝ) (z : Cell) : ℝ :=
  Real.sqrt (normForm σ z)

/-- The elliptic C*-norm is non-negative everywhere.

    (σ : ℝ)                    [the parameter]
    (z : Cell)                 [a point]
    0 ≤ cstarNormNeg σ z       [the norm is never negative] -/
theorem cstarNormNeg_nonneg (σ : ℝ) (z : Cell) : 0 ≤ cstarNormNeg σ z := by
  unfold cstarNormNeg
  exact Real.sqrt_nonneg (normForm σ z)

/-- Below zero, the elliptic C*-norm vanishes only at the origin.

    {σ : ℝ} (hσ : σ < 0)       [σ is below zero]
    (z : Cell)                 [a point]
    cstarNormNeg σ z = 0 ↔ z = 0  [norm is zero exactly when the point is zero] -/
theorem cstarNormNeg_eq_zero_iff {σ : ℝ} (hσ : σ < 0) (z : Cell) :
    cstarNormNeg σ z = 0 ↔ z = 0 := by
  unfold cstarNormNeg
  have h_nn := normForm_nonneg_of_nonpos hσ.le z
  constructor
  · intro h
    have h_sq : (Real.sqrt (normForm σ z)) ^ 2 = 0 ^ 2 := by rw [h]
    rw [Real.sq_sqrt h_nn, zero_pow (by norm_num : 2 ≠ 0)] at h_sq
    exact (normForm_eq_zero_iff_of_neg hσ z).mp h_sq
  · rintro rfl
    have hz : normForm σ 0 = 0 := by simp [normForm]
    rw [hz, Real.sqrt_zero]

/-- Multiplicativity of the elliptic C*-norm.

    {σ : ℝ} (hσ : σ ≤ 0)       [σ is at zero or below]
    (z w : Cell)               [two points]
    mul                        [multiply two points together, the σ way]
    cstarNormNeg σ (mul σ z w) = cstarNormNeg σ z * cstarNormNeg σ w
                               [norm of product is product of norms] -/
theorem cstarNormNeg_mul {σ : ℝ} (hσ : σ ≤ 0) (z w : Cell) :
    cstarNormNeg σ (mul σ z w) = cstarNormNeg σ z * cstarNormNeg σ w := by
  unfold cstarNormNeg
  rw [normForm_mul]
  exact Real.sqrt_mul (normForm_nonneg_of_nonpos hσ z) (normForm σ w)

/-- Conjugation preserves the elliptic C*-norm.

    (σ : ℝ) (z : Cell)         [the parameter and a point]
    cstarNormNeg σ (conj z) = cstarNormNeg σ z  [conjugate has same norm] -/
theorem cstarNormNeg_conj (σ : ℝ) (z : Cell) :
    cstarNormNeg σ (conj z) = cstarNormNeg σ z := by
  unfold cstarNormNeg
  rw [normForm_conj]

/-- The C*-identity holds under canonical conjugation for σ ≤ 0.

    {σ : ℝ} (hσ : σ ≤ 0)       [σ is at zero or below]
    (z : Cell)                 [a point]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    cstarNormNeg σ (mul σ (conj z) z) = (cstarNormNeg σ z) ^ 2
                               [norm of z* z is norm of z squared] -/
theorem cstar_identity_neg {σ : ℝ} (hσ : σ ≤ 0) (z : Cell) :
    cstarNormNeg σ (mul σ (conj z) z) = (cstarNormNeg σ z) ^ 2 := by
  rw [cstarNormNeg_mul hσ, cstarNormNeg_conj, sq]

/-- In the elliptic real C*-algebra, 1 + z* z is invertible for all z at σ < 0.

    {σ : ℝ} (hσ : σ < 0)       [σ is below zero]
    (z : Cell)                 [a point: two real numbers written side by side, a + b·x]
    one                        [the point (1, 0), which changes nothing when you multiply by it]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    ∃ w : Cell                 [there is at least one]
    mul σ (one + mul σ (conj z) z) w = one
                               [1 + z* z has a multiplicative inverse] -/
theorem one_add_star_mul_self_isUnit_neg {σ : ℝ} (hσ : σ < 0) (z : Cell) :
    ∃ w : Cell, mul σ (one + mul σ (conj z) z) w = one := by
  set v := one + mul σ (conj z) z
  have h_mul : mul σ (conj z) z = (normForm σ z, 0) := by
    rw [mul_comm, mul_conj]
  have hv : v = (1 + normForm σ z, 0) := by
    subst v
    ext <;> simp [one, h_mul]
  have hnorm : normForm σ v = (1 + normForm σ z) ^ 2 := by
    rw [hv]
    unfold normForm
    ring
  have h_nn : 0 ≤ normForm σ z := normForm_nonneg_of_nonpos hσ.le z
  have h_pos : 0 < 1 + normForm σ z := by linarith
  have hnorm_pos : 0 < normForm σ v := by
    rw [hnorm]
    exact sq_pos_of_pos h_pos
  have hnorm_ne : normForm σ v ≠ 0 := ne_of_gt hnorm_pos
  exact (isUnit_iff_normForm_ne_zero σ v).mpr hnorm_ne

-- ═══════════════════════════════════════════════════════════════════
-- §6. UNIFIED TRICHOTOMY C*-CLASSIFICATION
-- ═══════════════════════════════════════════════════════════════════

/-- The Unified C*-Classification across the σ-trichotomy:
    1. Elliptic (σ < 0): canonical conjugation satisfies the C*-identity.
    2. Hyperbolic (0 < σ): bifurcation into a real C*-algebra under starDiag
       and an indefinite Krein involution under conj which fails the C*-identity.
    3. Parabolic (σ = 0): non-trivial nilpotent radical prevents any C*-norm.

    (σ : ℝ)                    [the parameter]
    (z : Cell)                 [a point: two real numbers written side by side, a + b·x]
    Cell                       [a point: two real numbers written side by side, a + b·x]
    conj                       [the same point with its second slot negated]
    mul                        [multiply two points together, the σ way]
    ¬                          [this is not true]
    ∀                          [for every single one]
    ≠                          [is not equal to]
    ∧                          [both are true]
    ↔                          [exactly when, true in both directions]
    <                          [is smaller than]
    →                          [takes the left and gives the right]
                               [between statements, if the left holds so does the right]
    ∃                          [there is at least one]
    (σ < 0 → ∀ z : Cell, cstarNormNeg σ (mul σ (conj z) z) = (cstarNormNeg σ z) ^ 2)
                               [elliptic branch satisfies C*-identity under conj] -/
theorem trichotomy_cstar_classification (σ : ℝ) :
    (σ < 0 → ∀ z : Cell, cstarNormNeg σ (mul σ (conj z) z) = (cstarNormNeg σ z) ^ 2) ∧
    (0 < σ →
      (∀ z : Cell, cstarNormPos σ (mul σ (starDiag z) z) = (cstarNormPos σ z) ^ 2) ∧
      (¬ ∀ z : Cell, cstarNormPos σ (mul σ (conj z) z) = (cstarNormPos σ z) ^ 2)) ∧
    (σ = 0 →
      (∃ z : Cell, z ≠ 0 ∧ mul 0 z z = 0) ∧
      (∀ norm : Cell → ℝ,
        (∀ z : Cell, norm (mul 0 (conj z) z) = (norm z) ^ 2) →
        (∀ z : Cell, norm z = 0 ↔ z = 0) → False)) := by
  refine ⟨fun hσ z => cstar_identity_neg hσ.le z,
          fun hσ => ⟨cstar_identity_pos σ hσ, conj_not_cstar_pos σ hσ⟩,
          fun hσ => ?_⟩
  subst hσ
  exact ⟨jacobson_radical_zero, no_cstar_norm_zero⟩

end Trichotomy
end Apeiron
